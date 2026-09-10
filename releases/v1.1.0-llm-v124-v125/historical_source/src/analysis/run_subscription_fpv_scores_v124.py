"""Reuse v124 prompt/score contract through subscription SDK workers.

No fallback to third-party APIs. Each process gets only its entity packet.
Keep all predictions immutable before any cost/inventory/result evaluation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess

from src.analysis.fpv_llm_v124 import ROOT, OUT, FIELDS, build_entity, validate_response
from src.analysis.run_direct_fpv_scores_v124 import build_request, PROMPT_VERSION

WORKER = ROOT / 'src/analysis/fpv_subscription_worker.py'
SDK_PYTHON = ROOT / '.venv-llm-sdk/bin/python'
ADAPTER_VERSION = 'subscription-separated-v2'


def parse_model_json(text):
    """Allow one complete Markdown JSON wrapper, never repair values or prose."""
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, flags=re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def schema_for(dimension):
    return dict(type='object', additionalProperties=False, required=list(FIELDS[dimension]),
        properties={field: dict(type='object', additionalProperties=False,
            required=['score','confidence','reason'], properties={
                'score':dict(type=['number','null'], minimum=lo, maximum=hi),
                'confidence':dict(type='string', enum=['low','medium','high','unknown']),
                'reason':dict(type='string')}) for field,(lo,hi) in FIELDS[dimension].items()})


def run_worker(job, timeout):
    process = subprocess.Popen([str(SDK_PYTHON), str(WORKER)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        output, error = process.communicate(json.dumps(job), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        return dict(status='failed', error_kind='process_timeout')
    try:
        result = json.loads(output)
    except ValueError:
        result = dict(status='failed', error_kind='invalid_worker_receipt')
    result['worker_exit_code'] = process.returncode
    result['stderr_sha256'] = hashlib.sha256(error.encode()).hexdigest()
    # SDK process diagnostics are not scored or published as model responses.
    if process.returncode and result.get('status') == 'ok':
        result.update(status='failed', error_kind='nonzero_worker_exit')
    return result


def validate_prediction(record, entity):
    if record['entity_id'] != build_entity(entity['dimension'], entity['context'])['entity_id']:
        raise ValueError('Entity/context mismatch')
    if record.get('status') != 'ok':
        return
    response = record['response']
    if response.get('status') != 'ok' or response.get('active_tool_items') or response.get('turns') != 1:
        raise ValueError('Worker/tool/turn policy failed')
    if record['job']['model'] is not None and response.get('model') != record['job']['model']:
        raise ValueError('Model changed')
    if not isinstance(response.get('model'), str) or not response['model']:
        raise ValueError('Missing model identity')
    parsed = parse_model_json(response['final_response'])
    if validate_response(entity['dimension'], parsed) != record['scores']:
        raise ValueError('Saved scores differ from raw model response')


def run_one(entity, provider, model, repeat, directory, timeout, environment):
    request = build_request(entity, model)
    job = dict(provider=provider, model=model, effort='high', policy=request['messages'][0]['content'],
               prompt=request['messages'][1]['content'], schema=schema_for(entity['dimension']))
    signature = digest(dict(job=job, repeat=repeat, entity=entity, environment=environment,
                            prompt_version=PROMPT_VERSION, adapter_version=ADAPTER_VERSION))
    path = directory / (signature + '.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('signature') != signature or saved.get('job') != job:
            raise ValueError('Cache contract changed')
        validate_prediction(saved, entity)
        return saved
    response = run_worker(job, timeout)
    record = dict(signature=signature, entity_id=entity['entity_id'], dimension=entity['dimension'],
        context=entity['context'], provider=provider, requested_model=model, repeat=repeat,
        job=job, environment=environment, response=response, status='failed',
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_mode='direct_pretrained_judgment_no_retrieval', prompt_version=PROMPT_VERSION,
        adapter_version=ADAPTER_VERSION)
    if response.get('status') == 'ok':
        try:
            record['scores'] = validate_response(entity['dimension'], parse_model_json(response['final_response']))
            record['status'] = 'ok'
            validate_prediction(record, entity)
        except (ValueError, TypeError, KeyError) as error:
            record.update(status='failed', validation_error=type(error).__name__)
            record.pop('scores', None)
    # Atomic visibility without overwriting prior predictions, even on reruns.
    pending = path.with_suffix('.pending')
    with pending.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.link(pending, path)
    pending.unlink()
    return record


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--providers', nargs='+', choices=['codex','claude'], default=['codex','claude'])
    parser.add_argument('--codex-model', default='gpt-6-astra')
    parser.add_argument('--claude-model', default=None)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--max-calls', type=int, default=60)
    args = parser.parse_args()
    if not args.run_id.isalnum() or not 1 <= args.workers <= 4 or not 1 <= args.repeats <= 3:
        raise ValueError('Invalid run budget')
    if len(set(args.providers)) != len(args.providers) or not 10 <= args.timeout <= 180:
        raise ValueError('Invalid providers/timeout')
    path = OUT / 'data/pilot_entities.json'
    entities = json.loads(path.read_text())
    if len({e['entity_id'] for e in entities}) != len(entities) or not entities:
        raise ValueError('Empty/duplicated entities')
    calls = len(entities) * len(args.providers) * args.repeats
    if calls > args.max_calls or args.max_calls > 128:
        raise ValueError('Pilot budget exceeded')
    environment = dict(worker_sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256((ROOT/'src/analysis/fpv_llm_v124.py').read_bytes()).hexdigest(),
        prompt_source_sha256=hashlib.sha256((ROOT/'src/analysis/run_direct_fpv_scores_v124.py').read_bytes()).hexdigest(),
        entities_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for executable in ['codex','claude']:
        environment[executable+'_runtime'] = subprocess.run([executable,'--version'], check=True,
            capture_output=True, text=True).stdout.strip()
    installed = subprocess.run([str(SDK_PYTHON), '-c',
        "import importlib.metadata,json;print(json.dumps({x:importlib.metadata.version(x) for x in ['openai-codex','claude-agent-sdk']}))"],
        check=True, capture_output=True, text=True)
    environment['sdk_versions'] = json.loads(installed.stdout)
    directory = OUT / 'raw_subscription' / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'run.json'
    contract = dict(entities=entities, providers=args.providers, repeats=args.repeats,
        codex_model=args.codex_model, claude_model=args.claude_model, environment=environment,
        timeout=args.timeout, planned_calls=calls)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise ValueError('Run contract changed: use a new run-id')
    if not manifest_path.exists():
        with manifest_path.open('x') as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)
    records, model_by_provider = [], {}
    for provider in args.providers:
        model = getattr(args, provider+'_model')
        row = run_one(entities[0], provider, model, 0, directory, args.timeout, environment)
        records.append(row)
        print('preflight', provider, row['status'], flush=True)
        if row['status'] == 'ok':
            model_by_provider[provider] = row['response']['model']
    if len(model_by_provider) == len(args.providers):
        jobs = [(e,p,model_by_provider[p],r) for e in entities for p in args.providers
                for r in range(args.repeats) if not (e == entities[0] and r == 0)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one,*job,directory,args.timeout,environment) for job in jobs]
            for future in as_completed(futures):
                row = future.result(); records.append(row)
                print(len(records),'/',calls,row['provider'],row['dimension'],row['status'],flush=True)
    keys = [(r['entity_id'],r['provider'],r['repeat']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate prediction key')
    references = [dict(path=str((directory/(r['signature']+'.json')).relative_to(ROOT)),
        sha256=hashlib.sha256((directory/(r['signature']+'.json')).read_bytes()).hexdigest()) for r in records]
    success = sum(r['status'] == 'ok' for r in records)
    report = dict(status='predictions_complete_pending_evaluation' if success == calls else 'incomplete_predictions',
        planned_calls=calls, attempted_or_cached_calls=len(records), successful_calls=success,
        models=model_by_provider, environment=environment, responses=references,
        prediction_barrier_closed=success == calls, evaluated=False,
        limitations=['Judgments are unvalidated; tools absent in observed events does not prove empty tool exposure.',
                     'SDK decoding differs from legacy API; no temperature equivalence claim.'])
    (OUT/'reports'/(args.run_id+'_subscription_calls.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'responses','environment'}},indent=2))
    if success != calls:
        raise SystemExit(2)


if __name__ == '__main__':
    run_main()
