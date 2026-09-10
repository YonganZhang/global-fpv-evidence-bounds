"""Versioned, bounded recovery: quarantine exhausted draws, never reroll scores.

The original run and inference adapter remain immutable. Only two consecutive
unanswered process timeouts authorize a third draw. Missing raw is uncertainty,
not a timeout, and stops execution. Partial predictions never open evaluation.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from src.analysis import parallel_lean_dispatch_v124 as parallel
from src.analysis.prepare_global_scores_v124 import freeze_json, sha
from src.analysis.run_subscription_fpv_scores_v124 import digest

base = parallel.base
SCHEMA = 'fpv-lean-timeout-recovery-v1'
ANCHOR_NAME = '20260909T062512142279.json'
RUN_ID = 'recovery01'


def anchor_path():
    return base.LEAN / 'runs/lean01/receipts' / ANCHOR_NAME


def ref(path):
    return dict(path=str(path.relative_to(base.ROOT)), sha256=sha(path))


def read(path):
    return json.loads(path.read_text())


def strict_timeout(row):
    return base.unanswered(row) and row['response'].get('error_kind') == 'process_timeout'


def eligible(history):
    if not history:
        return True
    if history[-1]['status'] == 'ok':
        return False
    if len(history) == 1:
        return base.unanswered(history[0]) or base.format_retry_allowed(history[0])
    return len(history) == 2 and all(strict_timeout(row) for row in history)


def histories(source, rows):
    grouped = {}
    for r in source.get('rejected_responses', []) + source['responses']:
        row = rows[r['path']]
        history = grouped.setdefault(base.key(row), [])
        if not eligible(history):
            raise ValueError('Unauthorized inherited retry')
        history.append(row)
    return grouped


def apply_response(r, row, state):
    current, refs, rejected, grouped = state
    slot = base.key(row)
    history = grouped.setdefault(slot, [])
    if not eligible(history):
        raise ValueError('Unauthorized retry: accepted response or exhausted attempt cap')
    if slot in current:
        rejected.append(refs[slot])
    history.append(row)
    current[slot], refs[slot] = row, r


def initial_state(source, rows):
    grouped = histories(source, rows)
    refs = {base.key(rows[r['path']]): r for r in source['responses']}
    if len(refs) != len(source['responses']):
        raise ValueError('Duplicate inherited current slot')
    return ({k: rows[r['path']] for k, r in refs.items()}, refs,
            list(source.get('rejected_responses', [])), grouped)


def select_jobs(plan, lookup, state):
    # Eligible repairs first, with exhausted failures filtered out entirely.
    grouped = state[3]
    slots = [base.key(s) for s in plan['required_slots']]
    slots.sort(key=lambda k: not bool(grouped.get(k)))
    return [(lookup[k[0]], k[1], base.MODELS[k[1]], k[2])
            for k in slots if eligible(grouped.get(k, []))]


def parent_manifest():
    anchor = anchor_path()
    base.require_latest_source(anchor)
    directory = anchor.parent.parent
    if list(directory.rglob('*.pending')):
        raise ValueError('Parent pending raw: outcome uncertain')
    paths = {directory / 'contract.json', anchor}
    for name in ('receipts', 'dispatch_contracts', 'dispatch_intents', 'dispatch_events',
                 'predictions', 'runtime_retry'):
        paths.update((directory / name).rglob('*.json'))
    source = read(anchor)
    recorded = {r['path'] for r in source['responses'] + source.get('rejected_responses', [])}
    parent_raw = {str(p.relative_to(base.ROOT)) for name in ('predictions', 'runtime_retry')
                  for p in (directory / name).rglob('*.json')}
    if not parent_raw <= recorded:
        raise ValueError('Parent orphan raw outcome: no automatic replay')
    paths.update(base.ROOT / r['path'] for r in source['responses'] + source.get('rejected_responses', []))
    # The anchor also inherits the old full-protocol campaign. Freeze its
    # contracts/history, including auxiliary draws excluded from lean scoring.
    legacy_receipt = base.ROOT / base.load_plan()['source_receipt']
    base.require_latest_source(legacy_receipt)
    legacy = legacy_receipt.parent.parent
    if list(legacy.rglob('*.pending')):
        raise ValueError('Legacy parent pending raw: outcome uncertain')
    paths.update((legacy / 'contract.json', legacy / 'continuation.json'))
    for name in ('receipts', 'predictions', 'runtime_retry'):
        paths.update((legacy / name).rglob('*.json'))
    for p in (legacy / 'receipts').glob('*.json'):
        value = read(p)
        paths.update(base.ROOT / r['path'] for r in value['responses'] + value.get('rejected_responses', []))
    return {str(p.relative_to(base.ROOT)): sha(p) for p in sorted(paths)}


def contract_value(plan, environment, manifest):
    return dict(schema_version=SCHEMA, run_id=RUN_ID, arm='codex_primary', models=base.MODELS,
        environment=environment, plan=plan, required_slots=plan['required_slots'],
        expected_predictions=plan['expected_predictions'], source_receipt=ref(anchor_path()),
        parent_manifest=manifest, source_code_sha256={n: sha(Path(base.__file__).parent / n)
            for n in base.CODE_FILES}, dispatcher_sha256=sha(Path(parallel.__file__)),
        recovery_sha256=sha(Path(__file__)), workers=list(parallel.WORKERS), timeout_seconds=120,
        retry_policy='One ordinary eligible retry; third only after two unanswered process_timeout failures; never fourth.',
        failure_policy='Quarantine exhausted slots; uncertain dispatch stops; full frozen membership still required.',
        authorization='User approved controlled unanswered-timeout recovery and continuing other slots on 2026-09-09.')


def context(directory, initialize=False):
    if directory != base.LEAN / 'recovery' / RUN_ID:
        raise ValueError('Only the single authorized recovery01 successor is allowed')
    plan = base.load_plan()
    lookup = base._load_inputs(plan)
    anchor = anchor_path()
    manifest = parent_manifest()
    source = read(anchor)
    environment = read(anchor.parent.parent / 'contract.json')['environment']
    value = contract_value(plan, environment, manifest)
    path = directory / 'contract.json'
    if not path.exists():
        if not initialize:
            raise ValueError('Missing recovery contract')
        parallel.validate_receipt(anchor)
        # A completed event without raw must not become an eligible new draw.
        for p in anchor.parent.glob('*.json'):
            if any(e.get('execution_error') or not e.get('raw_ref')
                   for e in read(p).get('dispatch_events', [])):
                raise ValueError('Parent uncertain dispatch must be reconciled manually')
        freeze_json(path, value)
    if read(path) != value:
        raise ValueError('Recovery contract, parent history, model, environment or code changed')
    return plan, lookup, environment, source


def verify_events(intent_path, receipt, new_refs, rows):
    intent = read(intent_path)
    parallel.bounded(intent['requested_workers'], intent['budget'])
    events = parallel._events(intent_path)
    slots = [base.key(e) for e in events]
    by_slot = {base.key(rows[r['path']]): r for r in new_refs}
    if len(set(slots)) != len(slots) or len(by_slot) != len(new_refs) or set(slots) != set(by_slot):
        raise ValueError('Dispatch/raw bijection violated; unknown call cannot be replayed')
    for e in events:
        e['raw_ref'] = by_slot[base.key(e)]
        raw_path = base.ROOT / e['raw_ref']['path']
        destination = intent_path.parent.parent / 'predictions' / intent_path.stem / f"{e['index']:03d}"
        if raw_path.parent != destination or not e['end_ref'] or e['execution_error']:
            raise ValueError('Uncertain execution or unexpected attempt path')
        if (e['status'] != rows[e['raw_ref']['path']]['status'] or
                not e['dispatched_ns'] <= e['started_ns'] <= e['finished_ns']):
            raise ValueError('Invalid dispatch status or timing')
        overlap = sum(o['started_ns'] <= e['started_ns'] < o['finished_ns'] for o in events
                      if o['finished_ns'] is not None)
        if not 1 <= e['active_at_start'] == overlap <= intent['requested_workers']:
            raise ValueError('Invalid actual concurrency')
    failure_ends = [e['finished_ns'] for e in events if e['status'] != 'ok']
    if failure_ends and any(e['dispatched_ns'] > min(failure_ends) for e in events):
        raise ValueError('Dispatch continued after a failure')
    if len(events) > intent['budget'] or not events:
        raise ValueError('Invalid dispatch call budget or no progress')
    if (receipt.get('dispatch_events') != events or receipt.get('new_calls_this_invocation') != len(events)
            or receipt.get('actual_peak') != max(e['active_at_start'] for e in events)):
        raise ValueError('Dispatch metadata/count changed')
    return events


def validate_history(directory, ctx):
    plan, lookup, env, source = ctx
    expected = base.required_membership(plan)[0]
    paths = sorted((directory / 'receipts').glob('*.json'))
    intents = sorted((directory / 'dispatch_intents').glob('*.json'))
    if [p.name for p in paths] != [p.name for p in intents] or list(directory.rglob('*.pending')):
        raise ValueError('Incomplete invocation or pending raw: no automatic replay')
    inherited = source['responses'] + source.get('rejected_responses', [])
    own = [ref(p) for p in sorted((directory / 'predictions').rglob('*.json'))]
    rows = base.read_refs(inherited + own, lookup, env)
    if any(base.key(row) not in expected for row in rows.values()):
        raise ValueError('Unexpected required membership')
    for r in own:
        if rows[r['path']]['environment'] != env:
            raise ValueError('Recovery raw environment changed')
    state = initial_state(source, rows)
    supplied = {r['path']: r for r in inherited}
    previous = ref(anchor_path())
    reports = {}
    for path, ipath in zip(paths, intents):
        value, intent = read(path), read(ipath)
        base._check_receipt_hash(value)
        if (intent.get('contract') != ref(directory / 'contract.json') or
                intent.get('baseline_receipt') != previous or
                intent.get('receipt') != str(path.relative_to(base.ROOT)) or
                value.get('dispatch_intent') != ref(ipath) or
                value.get('contract_sha256') != sha(directory / 'contract.json')):
            raise ValueError('Changed contract or non-monotonic receipt parent')
        new_refs = value['raw_new_refs']
        if any(r['path'] in supplied or r not in own for r in new_refs):
            raise ValueError('Unknown or repeated raw attempt')
        verify_events(ipath, value, new_refs, rows)
        for r in new_refs:
            apply_response(r, rows[r['path']], state)
            supplied[r['path']] = r
        actual = base.summarize_state(plan, state[0])
        if (value['responses'] != list(state[1].values()) or value['rejected_responses'] != state[2]
                or any(value.get(k) != v for k, v in actual.items())
                or value.get('execution_error') is not None
                or value.get('global_result') is not False
                or value.get('scientific_accuracy_validated') is not False
                or value.get('source_receipt') != ref(anchor_path())):
            raise ValueError('Receipt lineage, membership, counts or result claim changed')
        previous = ref(path)
        reports[path] = actual
    if supplied != {r['path']: r for r in inherited + own}:
        raise ValueError('Orphan raw outcome: no automatic replay')
    event_dirs = {p.parent.name for p in (directory / 'dispatch_events').rglob('*.json')}
    if not event_dirs <= {p.stem for p in intents}:
        raise ValueError('Orphan dispatch event')
    return state, reports


def validate_receipt(receipt_path, require_complete=False):
    path = (base.ROOT / receipt_path).resolve()
    directory = base.LEAN / 'recovery' / RUN_ID
    if path.parent != directory / 'receipts':
        raise ValueError('Expected immutable recovery01 receipt')
    _, reports = validate_history(directory, context(directory))
    if path not in reports:
        raise ValueError('Receipt missing from immutable history')
    counts = reports[path]
    if require_complete and not counts['prediction_barrier_closed']:
        raise ValueError('Primary/audit predictions incomplete; evaluation remains blocked')
    return dict(all_integrity_checks_passed=True, **counts, counts=counts,
        receipt=str(path.relative_to(base.ROOT)), receipt_sha256=sha(path),
        scientific_accuracy_validated=False, global_result=False)


def status(directory, **values):
    path = directory / 'supervisor_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(observed_at=parallel.now(), **values), indent=2) + '\n')
    temporary.replace(path)


def run(args):
    if (any(type(v) is not int for v in (args.workers, args.max_new_calls, args.max_batches, args.batch_size)) or
            args.run_id != RUN_ID or args.workers not in parallel.WORKERS or
            not 0 <= args.max_new_calls <= 6000 or not 1 <= args.max_batches <= 128 or
            not 1 <= args.batch_size <= 128):
        raise ValueError('Invalid recovery run, workers or bounded budget')
    directory = base.LEAN / 'recovery' / RUN_ID
    directory.mkdir(parents=True, exist_ok=True)
    plan = base.load_plan()
    with ExitStack() as stack:
        for p in (base.GLOBAL / 'continuation_campaign.lock',
                  (base.ROOT / plan['source_receipt']).parent.parent / 'run.lock',
                  anchor_path().parent.parent / 'run.lock', directory / 'run.lock'):
            handle = stack.enter_context(p.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls = batches = no_success_batches = 0
        try:
            ctx = context(directory, initialize=True)
            plan, lookup, env, source = ctx
            if args.max_new_calls:
                runtime = base.environment()
                if any(runtime.get(k) != env.get(k) for k in base.INFERENCE_FIELDS):
                    raise ValueError('Original inference runtime changed')
            while True:
                # Parent remains frozen throughout the complete supervisor life.
                context(directory)
                state, reports = validate_history(directory, ctx)
                counts = base.summarize_state(plan, state[0])
                jobs = select_jobs(plan, lookup, state)
                if counts['prediction_barrier_closed']:
                    status(directory, state='predictions_complete_pending_evaluation', **counts)
                    return 0
                if calls >= args.max_new_calls or batches >= args.max_batches or not jobs:
                    status(directory, state='incomplete_budget_or_quarantined', calls=calls, batches=batches, **counts)
                    return 3
                budget = min(args.batch_size, args.max_new_calls - calls)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                ipath = directory / 'dispatch_intents' / (stamp + '.json')
                rpath = directory / 'receipts' / ipath.name
                previous = ref(max(reports)) if reports else ref(anchor_path())
                freeze_json(ipath, dict(schema_version=SCHEMA, contract=ref(directory / 'contract.json'),
                    baseline_receipt=previous, receipt=str(rpath.relative_to(base.ROOT)),
                    requested_workers=args.workers, budget=budget, started_at=parallel.now()))
                destinations = {base.key(dict(entity_id=j[0]['entity_id'], provider=j[1], repeat=j[3])):
                    directory / 'predictions' / stamp / f'{i:03d}' for i, j in enumerate(jobs[:budget])}

                def call(job):
                    dest = destinations[(job[0]['entity_id'], job[1], job[3])]
                    dest.mkdir(parents=True, exist_ok=False)
                    return base.run_one(*job, dest, 120, env)

                runner = parallel.Dispatch(args.workers, budget, call, directory / 'dispatch_events' / stamp)
                status(directory, state='running_batch', calls=calls, batches=batches,
                    workers=args.workers, intent=str(ipath.relative_to(base.ROOT)), **counts)
                runner.execute(jobs[:budget])
                events = parallel._events(ipath)
                own = [ref(p) for p in sorted((directory / 'predictions' / stamp).rglob('*.json'))]
                rows = base.read_refs(own, lookup, env)
                by_slot = {base.key(rows[r['path']]): r for r in own}
                for event in events:
                    event['raw_ref'] = by_slot.get(base.key(event))
                # Freeze an error receipt too; it deliberately cannot validate or resume.
                for r in own:
                    apply_response(r, rows[r['path']], state)
                actual = base.summarize_state(plan, state[0])
                errors = [e['execution_error'] for e in events if e['execution_error'] or not e['raw_ref']]
                execution_error = str(errors) if errors else None
                value = dict(schema_version=SCHEMA, run_id=RUN_ID, **actual,
                    responses=list(state[1].values()), rejected_responses=state[2],
                    contract_sha256=sha(directory / 'contract.json'), source_receipt=ref(anchor_path()),
                    dispatch_intent=ref(ipath), dispatch_events=events, raw_new_refs=own,
                    new_calls_this_invocation=len(events), actual_peak=runner.peak,
                    execution_error=execution_error, scientific_accuracy_validated=False, global_result=False)
                if execution_error:
                    value['prediction_barrier_closed'] = False
                value['receipt_sha256'] = digest(value)
                freeze_json(rpath, value)
                verify_events(ipath, value, own, rows)
                calls += len(events)
                batches += 1
                gained = actual['successful_predictions'] - counts['successful_predictions']
                no_success_batches = no_success_batches + 1 if gained == 0 else 0
                print(json.dumps(dict(receipt=str(rpath.relative_to(base.ROOT)), calls=calls, batches=batches, **actual)), flush=True)
                if no_success_batches >= 3:
                    raise ValueError('Three consecutive batches without accepted output; stop for review')
        except Exception as exc:
            status(directory, state='stopped_requires_review', reason=str(exc), calls=calls, batches=batches)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--workers', type=int, choices=parallel.WORKERS, default=2)
    parser.add_argument('--max-new-calls', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    raise SystemExit(run(parser.parse_args()))
