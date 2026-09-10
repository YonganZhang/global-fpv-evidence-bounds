"""Explicit operational parallelism for the already frozen lean science run.

After cutover use this entry for all worker counts. Interrupted invocations must
first be reconciled with --max-new-calls 0; uncertain calls are never replayed.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from threading import Lock
from time import monotonic_ns

from src.analysis import run_lean_global_scores_v124 as base
from src.analysis.prepare_global_scores_v124 import freeze_json, sha
from src.analysis.run_subscription_fpv_scores_v124 import digest

SCHEMA = 'fpv-lean-parallel-dispatch-v1'
WORKERS = (1, 2, 4, 8)


def now():
    return datetime.now(timezone.utc).isoformat()


def bounded(workers, budget):
    if type(workers) is not int or workers not in WORKERS or type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError('Parallel dispatch requires workers 1/2/4/8 and total budget 0..128')


class Dispatch:
    """One global budget and failure signal shared across all dispatch phases."""
    def __init__(self, workers, budget, call, event_dir):
        bounded(workers, budget)
        self.workers, self.budget, self.call = workers, budget, call
        self.event_dir = event_dir
        self.lock = Lock()
        self.events = []
        self.seen = set()
        self.active = self.peak = 0
        self.stopped = False

    def _invoke(self, job, index, dispatched_at, dispatched_ns):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            event = dict(index=index, entity_id=job[0]['entity_id'], provider=job[1], repeat=job[3],
                dispatched_at=dispatched_at, dispatched_ns=dispatched_ns,
                started_at=now(), started_ns=monotonic_ns(), active_at_start=self.active)
            freeze_json(self.event_dir / f'{index:03d}.start.json', event)
            self.events[index] = event
        status, error = None, None
        try:
            row = self.call(job)
            status = row['status']
            return row
        except Exception as exc:
            error = type(exc).__name__
            return None
        finally:
            with self.lock:
                end = dict(index=index, finished_at=now(), finished_ns=monotonic_ns(),
                    status=status, execution_error=error)
                # Failed calls publish the stop signal before another submit
                # can pass the same lock. Already dispatched futures drain.
                if status != 'ok' or error:
                    self.stopped = True
                self.active -= 1
                freeze_json(self.event_dir / f'{index:03d}.end.json', end)

    def execute(self, jobs, workers=None):
        phase_workers = workers or self.workers
        if not 1 <= phase_workers <= self.workers:
            raise ValueError('Phase exceeds authorized workers')
        iterator, results = iter(jobs), []
        with ThreadPoolExecutor(max_workers=phase_workers) as pool:
            active = set()

            def submit():
                with self.lock:
                    if self.stopped or len(self.events) >= self.budget:
                        return
                    job = next(iterator, None)
                    if job is None:
                        return
                    slot = job[0]['entity_id'], job[1], job[3]
                    if slot in self.seen:
                        raise ValueError('Duplicate dispatch slot')
                    self.seen.add(slot)
                    index = len(self.events)
                    self.events.append(None)
                    active.add(pool.submit(self._invoke, job, index, now(), monotonic_ns()))

            for _ in range(phase_workers):
                submit()
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                for _ in done:
                    submit()
        return results


def _relative(path):
    return str(path.relative_to(base.ROOT))


def _ref(path):
    return dict(path=_relative(path), sha256=sha(path))


def _read_ref(ref, parent):
    path = (base.ROOT / ref['path']).resolve()
    if path.parent != parent.resolve() or sha(path) != ref['sha256']:
        raise ValueError('Dispatch artifact path/hash changed')
    return json.loads(path.read_text())


def _operational(directory, anchor):
    return dict(schema_version=SCHEMA, user_authorized_parallelism=True,
        run_id=directory.name, base_contract=_ref(directory / 'contract.json'),
        anchor_receipt=_ref(anchor), allowed_workers=list(WORKERS), max_workers=8, max_calls=128,
        dispatcher_sha256=sha(Path(__file__)),
        base_code_sha256={n: sha(Path(base.__file__).parent / n) for n in base.CODE_FILES},
        inference_policy='Existing frozen environment/model/prompt and raw paths, no new science arm',
        cutover_policy='All receipts after anchor require operational metadata; use this entry for 1/2/4/8 workers')


def _events(intent_path):
    directory = intent_path.parent.parent / 'dispatch_events' / intent_path.stem
    starts = sorted(directory.glob('*.start.json'))
    events = []
    for path in starts:
        event = json.loads(path.read_text())
        end = path.with_name(path.name.replace('.start.json', '.end.json'))
        event['start_ref'] = _ref(path)
        if end.exists():
            event.update(json.loads(end.read_text()))
            event['end_ref'] = _ref(end)
        else:
            event.update(finished_at=None, finished_ns=None, status=None,
                execution_error='InterruptedDispatch', end_ref=None)
        events.append(event)
    if [e['index'] for e in events] != list(range(len(events))):
        raise ValueError('Missing dispatch start event')
    if len(list(directory.glob('*.end.json'))) != sum(e['end_ref'] is not None for e in events):
        raise ValueError('Orphan dispatch completion event')
    return events


def _verify_operational(directory, allow_pending=False):
    contract_path = directory / 'dispatch_contracts/parallel_v1.json'
    if not contract_path.exists():
        if list((directory / 'dispatch_intents').glob('*.json')) or any(
                json.loads(p.read_text()).get('dispatch_contract') for p in (directory / 'receipts').glob('*.json')):
            raise ValueError('Missing operational dispatch contract')
        return []
    contract = json.loads(contract_path.read_text())
    anchor = base.ROOT / contract['anchor_receipt']['path']
    if anchor.parent != directory / 'receipts' or contract != _operational(directory, anchor):
        raise ValueError('Operational contract/code/anchor changed')
    intents = sorted((directory / 'dispatch_intents').glob('*.json'))
    intent_by_receipt = {}
    pending = []
    for path in intents:
        intent = json.loads(path.read_text())
        if intent['dispatch_contract'] != _ref(contract_path):
            raise ValueError('Invocation changed operational contract')
        bounded(intent['requested_workers'], intent['budget'])
        previous = _read_ref(intent['baseline_receipt'], directory / 'receipts')
        if intent['baseline_raw_refs'] != previous['responses'] + previous.get('rejected_responses', []):
            raise ValueError('Dispatch baseline differs from preceding immutable receipt')
        receipt_path = base.ROOT / intent['receipt']
        if receipt_path.parent != directory / 'receipts' or receipt_path.name != path.name:
            raise ValueError('Invocation receipt identity changed')
        if not receipt_path.exists():
            pending.append(path)
            continue
        intent_by_receipt[receipt_path] = path
    for path in sorted((directory / 'receipts').glob('*.json')):
        if path <= anchor:
            continue
        if path not in intent_by_receipt:
            raise ValueError('Post-cutover receipt missing dispatch intent/metadata')
        receipt = json.loads(path.read_text())
        intent_path = intent_by_receipt[path]
        intent = json.loads(intent_path.read_text())
        if (receipt.get('dispatch_contract') != _ref(contract_path) or
                receipt.get('dispatch_intent') != _ref(intent_path)):
            raise ValueError('Dispatch metadata missing or changed')
        events = _events(intent_path)
        baseline = {r['path']: r for r in intent['baseline_raw_refs']}
        refs = {r['path']: r for r in receipt['responses'] + receipt.get('rejected_responses', [])}
        if any(refs.get(p) != r for p, r in baseline.items()):
            raise ValueError('Dispatch lost baseline attempt')
        new_refs = [r for p, r in refs.items() if p not in baseline]
        rows = {base.key(json.loads((base.ROOT / r['path']).read_text())): r for r in new_refs}
        slots = [base.key(e) for e in events]
        if len(set(slots)) != len(slots) or not set(rows) <= set(slots):
            raise ValueError('Duplicate dispatch or raw without dispatch event')
        for e in events:
            e['raw_ref'] = rows.get(base.key(e))
            if not 1 <= e['active_at_start'] <= intent['requested_workers']:
                raise ValueError('Observed workers exceed authorized request')
            if e['finished_ns'] is not None and e['finished_ns'] < e['started_ns']:
                raise ValueError('Invalid dispatch event order')
            if e['dispatched_ns'] > e['started_ns']:
                raise ValueError('Call started before dispatch')
            if e['status'] == 'ok' and not e['raw_ref']:
                raise ValueError('Successful call lacks frozen raw')
            if e['raw_ref'] and not e['execution_error']:
                row = json.loads((base.ROOT / e['raw_ref']['path']).read_text())
                if row['status'] != e['status']:
                    raise ValueError('Call status differs from persisted raw')
        if all(e['finished_ns'] is not None for e in events):
            for event in events:
                overlap = sum(e['started_ns'] <= event['started_ns'] < e['finished_ns'] for e in events)
                if event['active_at_start'] != overlap:
                    raise ValueError('Reported actual concurrency differs from timestamps')
        failures = [e['finished_ns'] for e in events if e['finished_ns'] is not None and
            (e['status'] != 'ok' or e['execution_error'])]
        if failures and any(e['dispatched_ns'] > min(failures) for e in events):
            raise ValueError('New dispatch after observed failure')
        peak = max((e['active_at_start'] for e in events), default=0)
        expected = dict(requested_workers=intent['requested_workers'], dispatch_budget=intent['budget'],
            actual_peak=peak, dispatch_events=events, raw_new_refs=sorted(new_refs, key=lambda r: r['path']),
            new_calls_this_invocation=len(events))
        if len(events) > intent['budget'] or any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('Dispatch events/counts/peak/budget changed')
        if any(e['execution_error'] for e in events) and not receipt.get('execution_error'):
            raise ValueError('Dispatch execution error omitted')
    if pending and not allow_pending:
        raise ValueError('Interrupted dispatch: reconcile with --max-new-calls 0 before new calls')
    return pending


def validate_receipt(receipt_path, require_complete=False):
    report = base.validate_receipt(receipt_path, require_complete=False)
    directory = (base.ROOT / receipt_path).resolve().parent.parent
    _verify_operational(directory)
    if require_complete and not report['prediction_barrier_closed']:
        raise ValueError('Primary/audit predictions incomplete; evaluation remains blocked')
    return {**report, 'parallel_dispatch_integrity_checked': True}


def _finish(intent_path, plan, source, lookup, env, interrupted=False):
    directory = intent_path.parent.parent
    intent = json.loads(intent_path.read_text())
    current, refs, rejected, rows = base._state(source, base._own_refs(directory), lookup, env,
        base.required_membership(plan)[0], directory)
    all_refs = list(refs.values()) + rejected
    baseline = {r['path']: r for r in intent['baseline_raw_refs']}
    new_refs = sorted((r for r in all_refs if r['path'] not in baseline), key=lambda r: r['path'])
    by_slot = {base.key(rows[r['path']]): r for r in new_refs}
    events = _events(intent_path)
    for event in events:
        event['raw_ref'] = by_slot.get(base.key(event))
    errors = [e['execution_error'] for e in events if e['execution_error']]
    execution_error = 'InterruptedDispatch' if interrupted else errors[0] if errors else None
    counts = base.summarize_state(plan, current)
    if execution_error:
        counts['prediction_barrier_closed'] = False
    summary = dict(schema_version=base.SCHEMA, run_id=directory.name, **counts,
        responses=list(refs.values()), rejected_responses=rejected,
        status='predictions_complete_pending_evaluation' if counts['prediction_barrier_closed'] else 'partial_predictions_not_global_result',
        source_receipt=plan['source_receipt'], source_receipt_sha256=plan['source_receipt_sha256'],
        contract_sha256=sha(directory / 'contract.json'), execution_error=execution_error,
        tracked_raw_attempts=len(rows), scientific_accuracy_validated=False, global_result=False,
        dispatch_contract=intent['dispatch_contract'], dispatch_intent=_ref(intent_path),
        requested_workers=intent['requested_workers'], dispatch_budget=intent['budget'],
        actual_peak=max((e['active_at_start'] for e in events), default=0),
        dispatch_events=events, raw_new_refs=new_refs, new_calls_this_invocation=len(events),
        observed_at=now(), stop_reason='execution_error' if execution_error else
            'failed_prediction_requires_review' if counts['failed_predictions'] else 'invocation_budget_or_complete')
    summary['receipt_sha256'] = digest(summary)
    path = base.ROOT / intent['receipt']
    freeze_json(path, summary)
    return path


def _status(path):
    temporary = path.parent.parent / 'status.pending'
    temporary.write_text(json.dumps({**json.loads(path.read_text()), 'receipt': _relative(path)}, indent=2) + '\n')
    temporary.replace(path.parent.parent / 'status.json')


def run(args):
    bounded(args.workers, args.max_new_calls)
    if not args.run_id.isalnum():
        raise ValueError('Invalid run ID')
    plan = base.load_plan()
    directory = base.LEAN / 'runs' / args.run_id
    source_directory = (base.ROOT / plan['source_receipt']).parent.parent
    if not (directory / 'contract.json').exists():
        raise ValueError('Parallel addon requires the initialized original lean run')
    with ExitStack() as stack:
        for path in [base.GLOBAL / 'continuation_campaign.lock', source_directory / 'run.lock', directory / 'run.lock']:
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = sorted((directory / 'receipts').glob('*.json'))[-1]
        base.validate_receipt(latest)
        lookup = base._load_inputs(plan)
        _, _, _, source = base._source(plan, lookup)
        contract = json.loads((directory / 'contract.json').read_text())
        env = contract['environment']
        if contract != base._contract(plan, env):
            raise ValueError('Frozen science contract changed')
        pending = _verify_operational(directory, allow_pending=True)
        if pending:
            if args.max_new_calls or len(pending) != 1:
                raise ValueError('Interrupted dispatch requires one zero-call reconciliation')
            latest = _finish(pending[0], plan, source, lookup, env, interrupted=True)
            report = validate_receipt(latest)
            _status(latest)
            print(json.dumps(report, indent=2))
            return report
        current, refs, rejected, rows = base._state(source, base._own_refs(directory), lookup, env,
            base.required_membership(plan)[0], directory)
        latest_refs = json.loads(latest.read_text())
        recorded = {r['path']: r for r in latest_refs['responses'] + latest_refs.get('rejected_responses', [])}
        if recorded != {r['path']: r for r in list(refs.values()) + rejected}:
            raise ValueError('Unreceipted raw without dispatch intent requires original recovery before cutover')
        if args.max_new_calls:
            pending_raw = [p for d in ('predictions', 'runtime_retry') for p in (directory / d).glob('*.pending')]
            if pending_raw:
                raise ValueError('Pending raw artifact requires explicit reconciliation; no SDK call is allowed')
            uncertain = [p for p in (directory / 'receipts').glob('*.json')
                if any(e.get('raw_ref') is None for e in json.loads(p.read_text()).get('dispatch_events', []))]
            if uncertain:
                raise ValueError('Prior dispatched call lacks atomic raw; outcome uncertain, no automatic replay')
            runtime = base.environment()
            if any(runtime.get(k) != env.get(k) for k in base.INFERENCE_FIELDS):
                raise ValueError('Original inference runtime changed')
        operational_path = directory / 'dispatch_contracts/parallel_v1.json'
        if not operational_path.exists():
            freeze_json(operational_path, _operational(directory, latest))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        intent_path = directory / 'dispatch_intents' / (stamp + '.json')
        intent = dict(schema_version=SCHEMA, dispatch_contract=_ref(operational_path),
            receipt=_relative(directory / 'receipts' / (stamp + '.json')), started_at=now(),
            requested_workers=args.workers, budget=args.max_new_calls,
            baseline_receipt=_ref(latest),
            baseline_raw_refs=latest_refs['responses'] + latest_refs.get('rejected_responses', []))
        freeze_json(intent_path, intent)
        jobs = base.select_required_jobs(plan, lookup, current, rows, args.retry_unanswered, args.repair_formats)
        failed = {k for k, r in current.items() if r['status'] != 'ok'}
        repairs = [j for j in jobs if (j[0]['entity_id'], j[1], j[3]) in failed]
        fresh = [j for j in jobs if (j[0]['entity_id'], j[1], j[3]) not in failed]

        def call(job):
            slot = job[0]['entity_id'], job[1], job[3]
            destination = base.destination_for(directory, slot, current)
            destination.mkdir(exist_ok=True)
            return base.run_one(*job, destination, 120, env)

        dispatch = Dispatch(args.workers, args.max_new_calls, call, directory / 'dispatch_events' / stamp)
        if len(repairs) == len(failed):
            repaired = dispatch.execute(repairs, workers=1)
            if len(repaired) == len(repairs) and not dispatch.stopped:
                for provider in base.MODELS:
                    job = next((j for j in fresh if j[1] == provider), None)
                    if job is not None:
                        dispatch.execute([job], workers=1)
                        fresh.remove(job)
                    if dispatch.stopped:
                        break
                if not dispatch.stopped:
                    dispatch.execute(fresh)
        latest = _finish(intent_path, plan, source, lookup, env)
        report = validate_receipt(latest)
        _status(latest)
        print(json.dumps(report, indent=2))
        summary = json.loads(latest.read_text())
        if summary['execution_error'] or summary['failed_predictions']:
            raise SystemExit(2)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-new-calls', type=int, default=0)
    parser.add_argument('--workers', type=int, choices=WORKERS, default=4)
    parser.add_argument('--repair-formats', action='store_true')
    parser.add_argument('--retry-unanswered', action='store_true')
    run(parser.parse_args())
