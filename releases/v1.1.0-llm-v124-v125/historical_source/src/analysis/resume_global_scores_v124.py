"""Continue a frozen global campaign without replacing successful draws.

Provider subsets affect scheduling only; both providers remain required for the
global barrier. One explicit retry is allowed for an unanswered runtime failure.
No prompt/provider runtime changes and no retry of null/low-confidence scores.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from threading import Lock

from src.analysis.fpv_llm_v124 import ROOT
from src.analysis.prepare_global_scores_v124 import GLOBAL, freeze_json, sha
from src.analysis.run_global_subscription_v124 import (
    MODELS, REPEATS, environment, validate_record, prompt_ordered_entity,
    key, execute_jobs, verify_history, format_retry_allowed)
from src.analysis.run_subscription_fpv_scores_v124 import run_one

INFERENCE_FIELDS=['worker_sha256','runner_sha256','contract_sha256','prompt_source_sha256',
                  'sdk_versions','codex_runtime','claude_runtime']


def unanswered(row):
    response=row.get('response',{})
    return (row.get('status')=='failed' and not row.get('scores') and not row.get('validation_error')
            and response.get('status')=='failed' and not response.get('final_response')
            and not response.get('active_tool_items')
            and response.get('error_kind') in {'RuntimeError','ResultError','process_timeout'})


def prediction_path(ref):
    path=(ROOT/ref['path']).resolve()
    if not path.is_relative_to(ROOT/'_outputs/v124') or path.suffix!='.json':
        raise ValueError('Prediction outside versioned output directory')
    return path


def require_source_inputs(source_contract,preparation_hash,entities_hash):
    if source_contract.get('preparation_sha256')!=preparation_hash:
        raise ValueError('Source and current global preparation differ')
    recorded=source_contract.get('entities_sha256',source_contract.get('environment',{}).get('entities_sha256'))
    if recorded!=entities_hash: raise ValueError('Source and current entity catalog differ')


def destination_for(directory,slot,current):
    return directory/('runtime_retry' if slot in current else 'predictions')


def require_unsealed(directory,new_calls):
    if new_calls and (directory/'continuation.json').exists():
        raise ValueError('Parent campaign is sealed; continue its child instead')


def require_latest_source(source_path):
    receipts=sorted(source_path.parent.glob('*.json'))
    if not receipts or source_path!=receipts[-1]:
        raise ValueError('Continuation must inherit the latest immutable receipt')


def read_refs(refs,lookup,env):
    rows={}
    for ref in refs:
        path=prediction_path(ref)
        if sha(path)!=ref['sha256']: raise ValueError('Frozen prediction changed')
        row=json.loads(path.read_text())
        if row.get('status') not in {'ok','failed'} or type(row.get('repeat')) is not int:
            raise ValueError('Invalid raw state or repeat')
        validate_record(row,lookup[row['entity_id']])
        if path.stem!=row['signature']: raise ValueError('Wrong prediction filename')
        if any(row['environment'].get(f)!=env.get(f) for f in INFERENCE_FIELDS):
            raise ValueError('Inference configuration changed')
        if ref['path'] in rows: raise ValueError('Duplicate raw reference')
        rows[ref['path']]=row
    return rows


def select_jobs(entities,current,all_rows,providers,retry_unanswered,repair_formats=False):
    attempts={}
    for row in all_rows.values(): attempts[key(row)]=attempts.get(key(row),0)+1
    jobs=[]
    for entity in entities:
        for repeat in range(REPEATS):
            for provider in MODELS:
                if provider not in providers: continue
                slot=entity['entity_id'],provider,repeat
                row=current.get(slot)
                if row is None or (attempts[slot]==1 and (
                    (retry_unanswered and unanswered(row)) or (repair_formats and format_retry_allowed(row)))):
                    jobs.append((entity,provider,MODELS[provider],repeat))
    return jobs


def receipt_state(source,own_refs,lookup,env):
    refs=source['responses'];rejected=list(source.get('rejected_responses',[]))
    all_refs={r['path']:r for r in refs+rejected}
    if len(all_refs)!=len(refs)+len(rejected): raise ValueError('Duplicate inherited reference')
    all_rows=read_refs(list(all_refs.values()),lookup,env)
    current={key(all_rows[r['path']]):all_rows[r['path']] for r in refs}
    current_refs={key(all_rows[r['path']]):r for r in refs}
    if len(current)!=len(refs): raise ValueError('Duplicate current draw')
    for ref in own_refs:
        rows=read_refs([ref],lookup,env);row=rows[ref['path']];slot=key(row)
        if slot in current:
            old=current[slot]
            prior_count=sum(key(v)==slot for v in all_rows.values())
            if not (unanswered(old) or format_retry_allowed(old)) or prior_count!=1:
                raise ValueError('Attempt to rescore success/null or exceed one runtime retry')
            rejected.append(current_refs[slot])
        if ref['path'] in all_rows: raise ValueError('Duplicate own raw')
        all_rows.update(rows);all_refs[ref['path']]=ref
        current[slot]=row;current_refs[slot]=ref
    return current,current_refs,rejected,all_rows


def run(args):
    if not args.run_id.isalnum() or not 0<=args.max_new_calls<=128 or not 1<=args.workers<=2:
        raise ValueError('Invalid run budget')
    if not args.providers or len(set(args.providers))!=len(args.providers): raise ValueError('Duplicate providers')
    source_path=(ROOT/args.source_receipt).resolve()
    if not source_path.is_relative_to(GLOBAL/'runs') or source_path.parent.name!='receipts':
        raise ValueError('Source must be an immutable global receipt')
    require_latest_source(source_path)
    source=json.loads(source_path.read_text())
    source_directory=source_path.parent.parent
    if source['contract_sha256']!=sha(source_directory/'contract.json'): raise ValueError('Source contract changed')
    source_contract=json.loads((source_directory/'contract.json').read_text())
    require_source_inputs(source_contract,sha(GLOBAL/'preparation.json'),sha(GLOBAL/'entities.json'))
    verify_history(source_directory,source['contract_sha256'])
    directory=GLOBAL/'runs'/args.run_id
    if directory==source_directory: raise ValueError('Continuation requires a distinct run ID')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_unsealed(directory,args.max_new_calls)
        entities=[prompt_ordered_entity(e) for e in json.loads((GLOBAL/'entities.json').read_text())]
        lookup={e['entity_id']:e for e in entities}
        preparation=json.loads((GLOBAL/'preparation.json').read_text())
        if len(lookup)!=len(entities): raise ValueError('Duplicate catalog entities')
        for name,h in preparation['artifact_hashes'].items():
            if sha(GLOBAL/name)!=h: raise ValueError('Global inputs changed')
        expected={(e['entity_id'],p,r) for e in entities for p in MODELS for r in range(REPEATS)}
        if source['expected_predictions']!=len(expected): raise ValueError('Changed global denominator')
        env={**environment(),'continuation_sha256':sha(Path(__file__))}
        contract=dict(schema_version='global-sdk-continuation-v1',models=MODELS,repeats=REPEATS,
            expected_predictions=len(expected),environment=env,source_receipt=str(source_path.relative_to(ROOT)),
            source_receipt_sha256=sha(source_path),preparation_sha256=sha(GLOBAL/'preparation.json'),
            retry_policy='At most one explicit unanswered-runtime or format retry per slot across inherited lineage; never rescore accepted null/low values.',
            provider_subset_policy='Scheduling only; both providers always required for global barrier.')
        freeze_json(directory/'contract.json',contract)
        # Seal the parent into a single continuation. A new run ID cannot reset
        # the unanswered retry budget by branching from the same old receipt.
        freeze_json(source_directory/'continuation.json',dict(run_id=args.run_id,
            source_receipt=str(source_path.relative_to(ROOT)),source_receipt_sha256=sha(source_path)))
        verify_history(directory,sha(directory/'contract.json'))
        raw_dir=directory/'predictions';raw_dir.mkdir(exist_ok=True)

        def state():
            paths=sorted(raw_dir.glob('*.json'))+sorted((directory/'runtime_retry').glob('*.json'))
            own=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths]
            for ref in own:
                row=json.loads((ROOT/ref['path']).read_text())
                if row['environment']!=env: raise ValueError('Current continuation environment changed')
            return receipt_state(source,own,lookup,env)

        current,refs,rejected,all_rows=state()
        jobs=select_jobs(entities,current,all_rows,set(args.providers),args.retry_unanswered,args.repair_formats)
        # Inherited unresolved errors in an unselected provider do not stop the
        # other provider; a fresh failure still stops this bounded invocation.
        new=[];execution_error=None
        call_count=0;counter_lock=Lock()
        def call(job):
            nonlocal call_count
            slot=job[0]['entity_id'],job[1],job[3]
            destination=destination_for(directory,slot,current)
            destination.mkdir(exist_ok=True)
            with counter_lock: call_count+=1
            return run_one(*job,destination,120,env)
        try:
            for provider in args.providers:
                job=next((j for j in jobs if j[1]==provider),None)
                if job is None or len(new)>=args.max_new_calls: continue
                row=call(job);new.append(row);jobs.remove(job)
                print('preflight',provider,row['status'],flush=True)
                if row['status']!='ok': break
            if all(r['status']=='ok' for r in new) and args.max_new_calls>len(new):
                new+=execute_jobs(jobs,call,args.workers,args.max_new_calls-len(new))
        except Exception as error:
            execution_error=type(error).__name__
        current,refs,rejected,all_rows=state()
        successful={k for k,r in current.items() if r['status']=='ok'}
        if not set(current)<=expected: raise ValueError('Unexpected global slots')
        failed=len(current)-len(successful)
        complete=successful==expected and not execution_error
        summary=dict(status='predictions_complete_pending_evaluation' if complete else 'partial_predictions_not_global_result',
            run_id=args.run_id,expected_predictions=len(expected),successful_predictions=len(successful),
            failed_predictions=failed,pending_predictions=len(expected)-len(current),
            selected_providers=args.providers,new_calls_this_invocation=call_count,
            inherited_successes=sum(r['status']=='ok' for r in read_refs(source['responses'],lookup,env).values()),
            tracked_raw_attempts=len(all_rows),execution_error=execution_error,
            prediction_barrier_closed=complete,global_result=False,scientific_accuracy_validated=False,
            by_provider={p:dict(successful=sum(k[1]==p for k in successful),expected=len(entities)*REPEATS) for p in MODELS},
            numeric_field_outputs=sum(v['score'] is not None for r in current.values() if r['status']=='ok' for v in r['scores'].values()),
            null_field_outputs=sum(v['score'] is None for r in current.values() if r['status']=='ok' for v in r['scores'].values()),
            responses=list(refs.values()),rejected_responses=rejected,contract_sha256=sha(directory/'contract.json'),
            stop_reason='new_failure' if any(r['status']!='ok' for r in new) else 'execution_error' if execution_error else 'invocation_budget_or_complete',
            observed_at=datetime.now(timezone.utc).isoformat())
        receipt=directory/'receipts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        freeze_json(receipt,summary)
        temporary=directory/'status.pending'
        temporary.write_text(json.dumps({**summary,'receipt':str(receipt.relative_to(ROOT))},indent=2)+'\n')
        temporary.replace(directory/'status.json')
        print(json.dumps({k:v for k,v in summary.items() if k not in {'responses','rejected_responses'}},indent=2))
        if execution_error or any(r['status']!='ok' for r in new): raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-receipt',required=True)
    parser.add_argument('--providers',nargs='+',choices=list(MODELS),default=list(MODELS))
    parser.add_argument('--max-new-calls',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--retry-unanswered',action='store_true')
    parser.add_argument('--repair-formats',action='store_true')
    args=parser.parse_args()
    # One active continuation for this global scoring campaign, across run IDs.
    with (GLOBAL/'continuation_campaign.lock').open('a') as campaign_lock:
        fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
