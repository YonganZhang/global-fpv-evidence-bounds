"""Bounded, resumable scheduler over the existing single-entity SDK adapter.

No new provider runtime; no result-driven prioritization. Raw records are frozen.
Any failure stops new scheduling and drains at most the already running worker.
A partial score campaign never unlocks the global numerical evaluator.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from src.analysis.fpv_llm_v124 import ROOT, OUT, build_entity
from src.analysis.prepare_global_scores_v124 import GLOBAL, freeze_json, sha, CONTEXT_FIELDS
from src.analysis.run_direct_fpv_scores_v124 import build_request, PROMPT_VERSION
from src.analysis.run_subscription_fpv_scores_v124 import run_one, validate_prediction, schema_for, WORKER, SDK_PYTHON, digest, ADAPTER_VERSION
from src.analysis.evaluate_subscription_pilot_v124 import load_predictions

MODELS = {'codex':'gpt-6-astra','claude':'claude-opus-5[1m]'}
REPEATS = 2


def prompt_ordered_entity(entity):
    # JSON manifests sort keys, but the established prompt serializes context in
    # insertion order. Restore the PILOT order without changing context values.
    fields=list(CONTEXT_FIELDS[entity['dimension']])
    if entity['dimension'] in {'country','province'}: fields.append('assessment_date')
    if set(entity['context'])!=set(fields): raise ValueError('Context fields changed')
    return {**entity,'context':{field:entity['context'][field] for field in fields}}


def key(record):
    return record['entity_id'],record['provider'],record['repeat']


def validate_record(record, entity):
    if record['context']!=entity['context'] or record['dimension']!=entity['dimension']:
        raise ValueError('Wrong context')
    if record['provider'] not in MODELS or record['repeat'] not in range(REPEATS):
        raise ValueError('Unexpected provider/repeat')
    model = MODELS[record['provider']]
    request = build_request(entity,model)
    job = record['job']
    if (job['policy']!=request['messages'][0]['content'] or job['prompt']!=request['messages'][1]['content']
        or job['schema']!=schema_for(entity['dimension']) or job['effort']!='high'
        or job['provider']!=record['provider'] or job['model'] not in {None,model}
        or record['prompt_version']!=PROMPT_VERSION):
        raise ValueError('Different scoring contract')
    validate_prediction(record,entity)
    expected=digest(dict(job=job,repeat=record['repeat'],entity=entity,environment=record['environment'],
        prompt_version=PROMPT_VERSION,adapter_version=ADAPTER_VERSION))
    if record['signature']!=expected or record['adapter_version']!=ADAPTER_VERSION:
        raise ValueError('Prediction signature changed')
    if record['status']=='ok' and record['response']['model']!=model:
        raise ValueError('Resolved model mismatch')


def environment():
    result = {name:sha(path) for name,path in {
        'worker_sha256':WORKER, 'runner_sha256':ROOT/'src/analysis/run_subscription_fpv_scores_v124.py',
        'scheduler_sha256':Path(__file__), 'contract_sha256':ROOT/'src/analysis/fpv_llm_v124.py',
        'prompt_source_sha256':ROOT/'src/analysis/run_direct_fpv_scores_v124.py',
        'entities_sha256':GLOBAL/'entities.json'}.items()}
    for cli in MODELS:
        result[cli+'_runtime']=subprocess.run([cli,'--version'],capture_output=True,text=True,check=True).stdout.strip()
    command="import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ['openai-codex','claude-agent-sdk']}))"
    result['sdk_versions']=json.loads(subprocess.run([str(SDK_PYTHON),'-c',command],check=True,capture_output=True,text=True).stdout)
    return result


def verify_history(directory, contract_hash):
    """A self-consistent rewrite of raw+scores must not replace frozen receipts."""
    hashes={}
    for receipt in sorted((directory/'receipts').glob('*.json')):
        old=json.loads(receipt.read_text())
        if old['contract_sha256']!=contract_hash:
            raise ValueError('Receipt belongs to a different contract')
        for ref in old['responses']+old.get('rejected_responses',[]):
            path=ref['path']
            if path in hashes and hashes[path]!=ref['sha256']:
                raise ValueError('Conflicting historical prediction hashes')
            hashes[path]=ref['sha256']
    for path,expected in hashes.items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Historical prediction changed after receipt')


def pending_jobs(entities, records):
    # Complete an entity's four draws before moving on; no energy/cost sorting.
    seen={key(r) for r in records}
    return [(e,p,MODELS[p],r) for e in entities for r in range(REPEATS) for p in MODELS
            if (e['entity_id'],p,r) not in seen]


def format_retry_allowed(row):
    response=row.get('response',{})
    return (row.get('status')=='failed' and bool(row.get('validation_error'))
            and response.get('status')=='ok' and response.get('turns')==1
            and not response.get('active_tool_items') and response.get('model')==MODELS[row['provider']])


def execute_jobs(jobs, call, workers, budget):
    """Bounded sliding window, not eager submission of 18k requests."""
    output=[]
    iterator=iter(jobs[:budget])
    stop=False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        def submit():
            job=next(iterator,None)
            if job is not None:
                active[pool.submit(call,job)]=job
        for _ in range(workers): submit()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                row=future.result()
                output.append(row)
                print('new',len(output),row['provider'],row['dimension'],row['status'],flush=True)
                if row['status']!='ok': stop=True
            if not stop:
                for _ in done: submit()
    return output


def run(args):
    if not args.run_id.isalnum() or not 0<=args.max_new_calls<=128 or not 1<=args.workers<=2:
        raise ValueError('Invalid bounded budget')
    directory=GLOBAL/'runs'/args.run_id
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entities=[prompt_ordered_entity(e) for e in json.loads((GLOBAL/'entities.json').read_text())]
        prepared=json.loads((GLOBAL/'preparation.json').read_text())
        for name,expected in prepared['artifact_hashes'].items():
            if sha(GLOBAL/name)!=expected: raise ValueError('Prepared input changed')
        lookup={e['entity_id']:e for e in entities}
        if len(lookup)!=len(entities) or not all(build_entity(e['dimension'],e['context'])==e for e in entities):
            raise ValueError('Invalid entity catalog')
        env=environment()
        seed,_,seed_hashes=load_predictions('sdkpilot02repaired')
        for row in seed:
            validate_record(row,lookup[row['entity_id']])
            for field in ['worker_sha256','runner_sha256','contract_sha256','prompt_source_sha256','sdk_versions','codex_runtime','claude_runtime']:
                if row['environment'][field]!=env[field]: raise ValueError('Pilot inference contract changed')
        seed_refs=[dict(path=p,sha256=h) for p,h in seed_hashes.items() if '/raw_subscription/' in p and not p.endswith('/run.json')]
        contract=dict(schema_version='global-single-entity-sdk-v1',models=MODELS,repeats=REPEATS,
            entities_sha256=sha(GLOBAL/'entities.json'),preparation_sha256=sha(GLOBAL/'preparation.json'),
            environment=env,seed_refs=seed_refs,seed_run='sdkpilot02repaired',expected_predictions=len(entities)*4,
            timeout=120,selection='Frozen context order, no outcome-dependent selection')
        freeze_json(directory/'contract.json',contract)
        verify_history(directory,sha(directory/'contract.json'))
        records=list(seed)
        refs=list(seed_refs)
        rejected=[]
        retry_calls=0
        for path in sorted((directory/'predictions').glob('*.json')):
            row=json.loads(path.read_text())
            validate_record(row,lookup[row['entity_id']])
            if path.stem!=row['signature']: raise ValueError('Prediction filename mismatch')
            if row['environment']!=env: raise ValueError('Resumed environment changed')
            retry_dir=directory/'format_retry'
            retry_path=retry_dir/path.name
            if retry_path.exists() and not format_retry_allowed(row):
                raise ValueError('Retry exists for an ineligible prediction')
            if format_retry_allowed(row) and (retry_path.exists() or (args.repair_formats and retry_calls<args.max_new_calls)):
                rejected.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                if not retry_path.exists():
                    retry_dir.mkdir(exist_ok=True)
                    retry_calls+=1
                    repaired=run_one(lookup[row['entity_id']],row['provider'],row['job']['model'],row['repeat'],retry_dir,120,env)
                    print('format_retry',row['provider'],repaired['status'],flush=True)
                else:
                    repaired=json.loads(retry_path.read_text())
                validate_record(repaired,lookup[row['entity_id']])
                if repaired['environment']!=env or key(repaired)!=key(row) or repaired['job']!=row['job'] or repaired['signature']!=path.stem:
                    raise ValueError('Format repair changed the draw contract')
                row=repaired;path=retry_path
            records.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        if len({key(r) for r in records})!=len(records): raise ValueError('Duplicate prediction key')
        jobs=pending_jobs(entities,records)
        new=[]
        remaining_budget=args.max_new_calls-retry_calls
        if all(r['status']=='ok' for r in records) and remaining_budget:
            (directory/'predictions').mkdir(exist_ok=True)
            def call(job):
                return run_one(*job,directory/'predictions',120,env)
            # Fresh provider checks precede the bulk on every invocation.
            for provider in MODELS:
                job=next((j for j in jobs if j[1]==provider),None)
                if job is None or len(new)>=remaining_budget: continue
                row=call(job);new.append(row)
                print('preflight',provider,row['status'],flush=True)
                jobs.remove(job)
                if row['status']!='ok': break
            if all(r['status']=='ok' for r in new):
                new+=execute_jobs(jobs,call,args.workers,remaining_budget-len(new))
            for row in new:
                validate_record(row,lookup[row['entity_id']])
                path=directory/'predictions'/(row['signature']+'.json')
                refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            records+=new
        succeeded=sum(r['status']=='ok' for r in records)
        failed=len(records)-succeeded
        numeric=sum(item['score'] is not None for r in records if r['status']=='ok' for item in r['scores'].values())
        nulls=sum(item['score'] is None for r in records if r['status']=='ok' for item in r['scores'].values())
        expected={(e['entity_id'],p,r) for e in entities for p in MODELS for r in range(REPEATS)}
        complete={key(r) for r in records if r['status']=='ok'}==expected and failed==0
        summary=dict(status='predictions_complete_pending_evaluation' if complete else 'partial_predictions_not_global_result',
            run_id=args.run_id,expected_predictions=len(entities)*4,seed_predictions=len(seed),
            attempted_predictions=len(records),successful_predictions=succeeded,failed_predictions=failed,
            pending_predictions=len(entities)*4-len(records),new_calls_this_invocation=len(new)+retry_calls,
            format_retries_this_invocation=retry_calls,total_attempts_including_rejected=len(records)+len(rejected),
            numeric_field_outputs=numeric,null_field_outputs=nulls,prediction_barrier_closed=complete,
            by_dimension={d:dict(expected=sum(e['dimension']==d for e in entities)*4,
                successful=sum(r['dimension']==d and r['status']=='ok' for r in records))
                for d in ['ecology','engineering','country','province']},
            global_result=False,scientific_accuracy_validated=False,contract_sha256=sha(directory/'contract.json'),
            responses=refs,rejected_responses=rejected,stop_reason='failed_prediction_requires_review' if failed else 'invocation_budget_or_complete',
            observed_at=datetime.now(timezone.utc).isoformat())
        receipt=directory/'receipts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        freeze_json(receipt,summary)
        # Latest is a replaceable pointer/summary; immutable receipts are truth.
        temporary=directory/'status.pending'
        temporary.write_text(json.dumps({**summary,'receipt':str(receipt.relative_to(ROOT))},indent=2)+'\n')
        temporary.replace(directory/'status.json')
        print(json.dumps({k:v for k,v in summary.items() if k!='responses'},indent=2))
        if failed: raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--max-new-calls',type=int,default=0,help='0 inspects/initializes, no SDK inference')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--repair-formats',action='store_true',help='One same-contract retry per schema failure; never retry null/low scores')
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.run_id.isalnum():
            freeze_json(GLOBAL/'runs'/args.run_id/'errors'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='execution_failed_not_complete',exception_type=type(error).__name__,global_result=False,
                     note='Prior status is historical; immutable raw records may need recovery. No exception text or credentials copied.'))
        raise
