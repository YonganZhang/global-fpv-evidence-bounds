"""Frozen Codex repeat-0 primary and Claude repeat-0 audit scheduler.

Reuse the original SDK adapter and prompt validation. Audit disagreement is not
accuracy and audit responses never enter a provider average. Validation is read
only and does not inspect installed SDKs or start provider processes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from threading import Lock

from src.analysis.fpv_llm_v124 import ROOT
from src.analysis.prepare_global_scores_v124 import GLOBAL, freeze_json, sha
from src.analysis.lean_global_plan_v124 import LEAN, load_plan
from src.analysis.resume_global_scores_v124 import (
    INFERENCE_FIELDS, destination_for, read_refs, receipt_state,
    require_latest_source, require_source_inputs, unanswered)
from src.analysis.run_global_subscription_v124 import (
    MODELS, REPEATS, environment, execute_jobs, format_retry_allowed, key,
    prompt_ordered_entity, verify_history)
from src.analysis.run_subscription_fpv_scores_v124 import digest, run_one

SCHEMA = 'fpv-primary-audit-run-v1'
CODE_FILES = ('run_lean_global_scores_v124.py', 'lean_global_plan_v124.py',
    'resume_global_scores_v124.py', 'run_global_subscription_v124.py',
    'run_subscription_fpv_scores_v124.py', 'fpv_subscription_worker.py',
    'fpv_llm_v124.py', 'run_direct_fpv_scores_v124.py', 'prepare_global_scores_v124.py',
    'validate_global_campaign_v124.py')


def required_membership(plan):
    """Reject role/provider substitutions even when the denominator matches."""
    slots = plan['required_slots']
    expected = {key(s) for s in slots}
    if len(expected) != len(slots):
        raise ValueError('Duplicate required slot')
    for s in slots:
        if type(s['repeat']) is not int or s['repeat'] != 0:
            raise ValueError('Only repeat zero is required')
        if (s['role'], s['provider']) not in {('primary', 'codex'), ('audit', 'claude')}:
            raise ValueError('Wrong primary/audit role')
    primary = {key(s) for s in slots if s['role'] == 'primary'}
    audit = expected - primary
    if (len(primary), len(audit), len(expected)) != (
            plan['primary_count'], plan['audit_count'], plan['expected_predictions']):
        raise ValueError('Required membership counts differ')
    if not {s[0] for s in audit} <= {s[0] for s in primary}:
        raise ValueError('Audit entity lacks primary slot')
    return expected, primary, audit


def select_required_jobs(plan, lookup, current, all_rows,
                         retry_unanswered=False, repair_formats=False):
    """Schedule frozen membership only; the combined retry cap is one."""
    required_membership(plan)
    attempts = {}
    for row in all_rows.values():
        attempts[key(row)] = attempts.get(key(row), 0) + 1
    jobs = []
    for slot in plan['required_slots']:
        k = key(slot)
        row = current.get(k)
        if row is None or (attempts.get(k) == 1 and (
                (retry_unanswered and unanswered(row)) or
                (repair_formats and format_retry_allowed(row)))):
            jobs.append((lookup[k[0]], k[1], MODELS[k[1]], k[2]))
    return jobs


def summarize_state(plan, current):
    expected, primary, audit = required_membership(plan)
    if not set(current) <= expected:
        raise ValueError('Unexpected required slots')
    successful = {k for k, row in current.items() if row['status'] == 'ok'}
    return dict(expected_predictions=len(expected), primary_expected=len(primary),
        audit_expected=len(audit), successful_predictions=len(successful),
        primary_successful=len(successful & primary), audit_successful=len(successful & audit),
        failed_predictions=len(current) - len(successful),
        pending_predictions=len(expected) - len(current),
        prediction_barrier_closed=successful == expected)


def _load_inputs(plan):
    required_membership(plan)
    if sha(GLOBAL / 'preparation.json') != plan['preparation_sha256']:
        raise ValueError('Preparation changed')
    if sha(GLOBAL / 'entities.json') != plan['entities_sha256']:
        raise ValueError('Entity catalog changed')
    prepared = json.loads((GLOBAL / 'preparation.json').read_text())
    for name, expected in prepared['artifact_hashes'].items():
        if sha(GLOBAL / name) != expected:
            raise ValueError('Prepared input changed')
    entities = [prompt_ordered_entity(e) for e in json.loads((GLOBAL / 'entities.json').read_text())]
    lookup = {e['entity_id']: e for e in entities}
    if len(lookup) != len(entities) or {s[0] for s in required_membership(plan)[1]} != set(lookup):
        raise ValueError('Primary membership differs from entity catalog')
    return lookup


def _source(plan, lookup):
    """Read-only equivalent of legacy source integrity checks, before filtering."""
    path = (ROOT / plan['source_receipt']).resolve()
    if not path.is_relative_to(GLOBAL / 'runs') or path.parent.name != 'receipts':
        raise ValueError('Expected immutable legacy source receipt')
    require_latest_source(path)
    if sha(path) != plan['source_receipt_sha256']:
        raise ValueError('Source receipt changed')
    source = json.loads(path.read_text())
    directory = path.parent.parent
    contract_path = directory / 'contract.json'
    contract = json.loads(contract_path.read_text())
    if sha(contract_path) != plan['source_contract_sha256'] or source['contract_sha256'] != sha(contract_path):
        raise ValueError('Source contract changed')
    require_source_inputs(contract, plan['preparation_sha256'], plan['entities_sha256'])
    verify_history(directory, sha(contract_path))
    refs = source['responses']
    rejected = source.get('rejected_responses', [])
    rows = read_refs(refs + rejected, lookup, contract['environment'])
    current = {key(rows[r['path']]): rows[r['path']] for r in refs}
    expected = {(eid, p, r) for eid in lookup for p in MODELS for r in range(REPEATS)}
    successful = {k for k, row in current.items() if row['status'] == 'ok'}
    if (len(current) != len(refs) or not set(current) <= expected or
            source['expected_predictions'] != len(expected) or
            source['successful_predictions'] != len(successful) or
            source['failed_predictions'] != len(current) - len(successful) or
            source['pending_predictions'] != len(expected) - len(current) or
            source['prediction_barrier_closed'] != (successful == expected) or
            source.get('global_result') is not False):
        raise ValueError('Invalid source receipt membership/counts')
    required = required_membership(plan)[0]
    filtered = dict(responses=[r for r in refs if key(rows[r['path']]) in required],
        rejected_responses=[r for r in rejected if key(rows[r['path']]) in required])
    return path, source, contract, filtered


def _own_refs(directory):
    paths = sorted((directory / 'predictions').glob('*.json')) + sorted((directory / 'runtime_retry').glob('*.json'))
    return [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths]


def _check_attempts(current, refs, rejected, all_rows, expected):
    if not set(current) <= expected or any(key(r) not in expected for r in all_rows.values()):
        raise ValueError('Unexpected primary/audit slot')
    if len(all_rows) != len(refs) + len(rejected):
        raise ValueError('Missing or duplicate attempt reference')
    grouped = {}
    prior_by_slot = {}
    for row in all_rows.values():
        grouped.setdefault(key(row), []).append(row)
    for ref in rejected:
        row = all_rows[ref['path']]
        prior_by_slot.setdefault(key(row), []).append(row)
    if set(grouped) != set(current):
        raise ValueError('Rejected attempt lacks its current response')
    for slot in current:
        attempts = grouped[slot]
        prior = prior_by_slot.get(slot, [])
        if len(attempts) not in {1, 2} or len(prior) != len(attempts) - 1:
            raise ValueError('More than one combined retry or malformed attempt history')
        if prior and not (unanswered(prior[0]) or format_retry_allowed(prior[0])):
            raise ValueError('Accepted/null or ineligible response was rerolled')


def _state(source, own_refs, lookup, env, expected, directory):
    for ref in own_refs:
        path = (ROOT / ref['path']).resolve()
        if path.parent not in {directory / 'predictions', directory / 'runtime_retry'}:
            raise ValueError('Raw response belongs to another run')
        row = json.loads(path.read_text())
        if row['environment'] != env:
            raise ValueError('Lean environment changed')
    state = receipt_state(source, own_refs, lookup, env)
    _check_attempts(*state, expected)
    return state


def _check_receipt_hash(receipt):
    value = dict(receipt)
    recorded = value.pop('receipt_sha256', None)
    if recorded != digest(value):
        raise ValueError('Receipt self-hash changed')


def _validate_snapshot(receipt, contract_hash, plan, source, lookup, env, directory):
    _check_receipt_hash(receipt)
    if receipt['contract_sha256'] != contract_hash:
        raise ValueError('Receipt belongs to another contract')
    refs = receipt['responses'] + receipt.get('rejected_responses', [])
    inherited = {r['path']: r for r in source['responses'] + source.get('rejected_responses', [])}
    supplied = {r['path']: r for r in refs}
    if len(supplied) != len(refs) or any(supplied.get(p) != r for p, r in inherited.items()):
        raise ValueError('Duplicate or missing inherited references')
    own = [r for r in refs if r['path'] not in inherited]
    # First attempts precede retries regardless of response-list insertion order.
    own.sort(key=lambda r: ((ROOT / r['path']).parent.name == 'runtime_retry', r['path']))
    current, current_refs, rejected, rows = _state(source, own, lookup, env,
        required_membership(plan)[0], directory)
    if ({r['path']: r for r in receipt['responses']} != {r['path']: r for r in current_refs.values()} or
            {r['path']: r for r in receipt.get('rejected_responses', [])} != {r['path']: r for r in rejected}):
        raise ValueError('Current/rejected attempts do not match lineage')
    actual = summarize_state(plan, current)
    if receipt.get('execution_error'):
        actual['prediction_barrier_closed'] = False
    if any(receipt.get(k) != v for k, v in actual.items()):
        raise ValueError('Receipt membership/counts/barrier mismatch')
    if (receipt.get('global_result') is not False or receipt.get('scientific_accuracy_validated') is not False or
            receipt.get('source_receipt') != plan['source_receipt'] or
            receipt.get('source_receipt_sha256') != plan['source_receipt_sha256']):
        raise ValueError('Invalid provenance or result claim')
    return actual, supplied


def _verify_snapshots(directory, contract_hash, plan, source, lookup, env):
    verify_history(directory, contract_hash)
    previous = {}
    reports = {}
    for path in sorted((directory / 'receipts').glob('*.json')):
        report, supplied = _validate_snapshot(json.loads(path.read_text()), contract_hash,
            plan, source, lookup, env, directory)
        if any(supplied.get(p) != r for p, r in previous.items()):
            raise ValueError('Receipt dropped or changed historical attempts')
        previous = supplied
        reports[path] = report
    return reports


def _seal(plan, run_id):
    return dict(schema_version=SCHEMA, run_id=run_id,
        successor_path=str((LEAN / 'runs' / run_id).relative_to(ROOT)),
        source_receipt=plan['source_receipt'], source_receipt_sha256=plan['source_receipt_sha256'])


def _contract(plan, env):
    return dict(schema_version=SCHEMA, user_authorized_primary_audit=True, arm='codex_primary',
        models=MODELS, repeats=1, expected_predictions=plan['expected_predictions'],
        primary_expected=plan['primary_count'], audit_expected=plan['audit_count'],
        required_slots=plan['required_slots'], plan=plan, environment=env,
        source_code_sha256={name: sha(Path(__file__).parent / name) for name in CODE_FILES},
        source_receipt=plan['source_receipt'], source_receipt_sha256=plan['source_receipt_sha256'],
        preparation_sha256=plan['preparation_sha256'], entities_sha256=plan['entities_sha256'],
        retry_policy='One combined explicit format/runtime retry across original lineage; accepted null/low never rerolled.',
        audit_policy='Claude repeat 0 is a score-blind stratified disagreement audit, not accuracy validation or a primary average.')


def validate_receipt(receipt_path, require_complete=False):
    """Return verified primary/audit coverage without writes or SDK processes."""
    path = (ROOT / receipt_path).resolve()
    if not path.is_relative_to(LEAN / 'runs') or path.parent.name != 'receipts':
        raise ValueError('Expected immutable lean receipt')
    plan = load_plan()
    lookup = _load_inputs(plan)
    source_path, _, source_contract, source = _source(plan, lookup)
    directory = path.parent.parent
    contract_path = directory / 'contract.json'
    contract = json.loads(contract_path.read_text())
    env = contract['environment']
    if contract != _contract(plan, env):
        raise ValueError('Lean contract differs from frozen plan')
    if (env.get('lean_runner_sha256') != sha(Path(__file__)) or
            env.get('lean_plan_sha256') != sha(ROOT / 'src/analysis/lean_global_plan_v124.py') or
            any(env.get(f) != source_contract['environment'].get(f) for f in INFERENCE_FIELDS)):
        raise ValueError('Frozen code or inference environment changed')
    if json.loads((source_path.parent.parent / 'continuation.json').read_text()) != _seal(plan, directory.name):
        raise ValueError('Run is not the single sealed successor')
    reports = _verify_snapshots(directory, sha(contract_path), plan, source, lookup, env)
    if path not in reports:
        raise ValueError('Receipt is absent from immutable history')
    report = dict(all_integrity_checks_passed=True, **reports[path],
        source_receipt=plan['source_receipt'], source_receipt_sha256=plan['source_receipt_sha256'],
        receipt=str(path.relative_to(ROOT)), receipt_sha256=sha(path),
        scientific_accuracy_validated=False, global_result=False)
    if require_complete and not report['prediction_barrier_closed']:
        raise ValueError('Primary/audit predictions incomplete; evaluation remains blocked')
    return report


def run(args):
    if not args.run_id.isalnum() or not 0 <= args.max_new_calls <= 128 or not 1 <= args.workers <= 2:
        raise ValueError('Invalid bounded run budget')
    plan = load_plan()
    lookup = _load_inputs(plan)
    source_path = (ROOT / plan['source_receipt']).resolve()
    source_directory = source_path.parent.parent
    directory = LEAN / 'runs' / args.run_id
    with ExitStack() as stack:
        # Shared continuation lock plus the source run lock also exclude the
        # legacy entry points that only acquire one of those locks.
        for lock_path in [GLOBAL / 'continuation_campaign.lock', source_directory / 'run.lock']:
            handle = stack.enter_context(lock_path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_path, original, source_contract, source = _source(plan, lookup)
        seal_path = source_directory / 'continuation.json'
        if seal_path.exists() and json.loads(seal_path.read_text()) != _seal(plan, args.run_id):
            raise ValueError('Source already has a different successor; branching is forbidden')
        env = {**environment(), 'lean_runner_sha256': sha(Path(__file__)),
            'lean_plan_sha256': sha(ROOT / 'src/analysis/lean_global_plan_v124.py')}
        if any(env.get(f) != source_contract['environment'].get(f) for f in INFERENCE_FIELDS):
            raise ValueError('Original inference environment changed')
        contract = _contract(plan, env)
        if not (directory / 'contract.json').exists():
            from src.analysis.validate_global_campaign_v124 import validate
            validate(plan['source_receipt'], False)
        directory.mkdir(parents=True, exist_ok=True)
        handle = stack.enter_context((directory / 'run.lock').open('a'))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Fully validate candidate state before committing the successor seal.
        if (directory / 'contract.json').exists() and json.loads((directory / 'contract.json').read_text()) != contract:
            raise ValueError('Frozen lean contract changed')
        expected = required_membership(plan)[0]
        current, refs, rejected, rows = _state(source, _own_refs(directory), lookup, env, expected, directory)
        if (directory / 'contract.json').exists():
            _verify_snapshots(directory, sha(directory / 'contract.json'), plan, source, lookup, env)
        freeze_json(directory / 'contract.json', contract)
        freeze_json(seal_path, _seal(plan, args.run_id))
        jobs = select_required_jobs(plan, lookup, current, rows, args.retry_unanswered, args.repair_formats)
        # Existing failures must be explicitly retried first. An unresolved or
        # exhausted failure prevents additional fresh draws in this campaign.
        failed_slots = {k for k, row in current.items() if row['status'] != 'ok'}
        repair_jobs = [j for j in jobs if (j[0]['entity_id'], j[1], j[3]) in failed_slots]
        fresh_jobs = [j for j in jobs if (j[0]['entity_id'], j[1], j[3]) not in failed_slots]
        blocked = len(repair_jobs) != len(failed_slots)
        new = []
        call_count = 0
        counter_lock = Lock()
        execution_error = None

        def call(job):
            nonlocal call_count
            slot = job[0]['entity_id'], job[1], job[3]
            destination = destination_for(directory, slot, current)
            destination.mkdir(exist_ok=True)
            with counter_lock:
                call_count += 1
            return run_one(*job, destination, 120, env)

        try:
            if not blocked:
                for job in repair_jobs:
                    if call_count >= args.max_new_calls:
                        break
                    row = call(job)
                    new.append(row)
                    if row['status'] != 'ok':
                        break
                repaired_all = len(new) == len(repair_jobs) and all(r['status'] == 'ok' for r in new)
                if repaired_all:
                    for provider in MODELS:
                        job = next((j for j in fresh_jobs if j[1] == provider), None)
                        if job is None or call_count >= args.max_new_calls:
                            continue
                        row = call(job)
                        new.append(row)
                        fresh_jobs.remove(job)
                        if row['status'] != 'ok':
                            break
                    if all(r['status'] == 'ok' for r in new):
                        new += execute_jobs(fresh_jobs, call, args.workers, args.max_new_calls - call_count)
        except Exception as error:
            execution_error = type(error).__name__
        current, refs, rejected, rows = _state(source, _own_refs(directory), lookup, env, expected, directory)
        counts = summarize_state(plan, current)
        if execution_error:
            counts['prediction_barrier_closed'] = False
        summary = dict(schema_version=SCHEMA, run_id=args.run_id, **counts,
            status='predictions_complete_pending_evaluation' if counts['prediction_barrier_closed'] else 'partial_predictions_not_global_result',
            responses=list(refs.values()), rejected_responses=rejected,
            source_receipt=plan['source_receipt'], source_receipt_sha256=plan['source_receipt_sha256'],
            source_auxiliary_current_predictions=len(original['responses']) - len(source['responses']),
            contract_sha256=sha(directory / 'contract.json'), new_calls_this_invocation=call_count,
            tracked_raw_attempts=len(rows), execution_error=execution_error,
            scientific_accuracy_validated=False, global_result=False,
            stop_reason='execution_error' if execution_error else 'failed_prediction_requires_review' if counts['failed_predictions'] else 'invocation_budget_or_complete',
            observed_at=datetime.now(timezone.utc).isoformat())
        summary['receipt_sha256'] = digest(summary)
        receipt = directory / 'receipts' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        freeze_json(receipt, summary)
        report = validate_receipt(receipt)
        temporary = directory / 'status.pending'
        temporary.write_text(json.dumps({**summary, 'receipt': str(receipt.relative_to(ROOT))}, indent=2) + '\n')
        temporary.replace(directory / 'status.json')
        print(json.dumps(report, indent=2))
        if execution_error or counts['failed_predictions']:
            raise SystemExit(2)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-new-calls', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--repair-formats', action='store_true')
    parser.add_argument('--retry-unanswered', action='store_true')
    run(parser.parse_args())
