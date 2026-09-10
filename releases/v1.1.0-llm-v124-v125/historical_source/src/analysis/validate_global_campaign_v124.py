"""Read frozen receipts and report scoring coverage; never call an SDK.

Incomplete coverage is a successful progress inspection, NOT successful global
evaluation. --require-complete fails unless all planned response slots exist.
Even complete formatted answers may contain explicit unknown (null) judgments.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT,FIELDS
from src.analysis.prepare_global_scores_v124 import GLOBAL,freeze_json,immutable_bytes,sha
from src.analysis.run_global_subscription_v124 import MODELS,REPEATS,key,prompt_ordered_entity,verify_history
from src.analysis.resume_global_scores_v124 import read_refs,require_source_inputs
from src.analysis.evaluate_subscription_pilot_v124 import summarize_scores


def require_full_membership(current,expected):
    if {key(r) for r in current if r['status']=='ok'}!=expected:
        raise ValueError('Global scores incomplete; final potential evaluation remains blocked')


def validate(receipt_path,require_complete=False):
    receipt_path=(ROOT/receipt_path).resolve()
    if not receipt_path.is_relative_to(GLOBAL/'runs') or receipt_path.parent.name!='receipts':
        raise ValueError('Expected immutable campaign receipt')
    directory=receipt_path.parent.parent
    receipt=json.loads(receipt_path.read_text())
    contract_path=directory/'contract.json'
    contract=json.loads(contract_path.read_text())
    if sha(contract_path)!=receipt['contract_sha256']: raise ValueError('Contract changed')
    verify_history(directory,sha(contract_path))
    require_source_inputs(contract,sha(GLOBAL/'preparation.json'),sha(GLOBAL/'entities.json'))
    entities=[prompt_ordered_entity(e) for e in json.loads((GLOBAL/'entities.json').read_text())]
    lookup={e['entity_id']:e for e in entities}
    expected={(e['entity_id'],p,r) for e in entities for p in MODELS for r in range(REPEATS)}
    refs=receipt['responses'];rejected=receipt.get('rejected_responses',[])
    all_rows=read_refs(refs+rejected,lookup,contract['environment'])
    current=[all_rows[r['path']] for r in refs]
    slots=[key(r) for r in current]
    accepted=[r for r in current if r['status']=='ok']
    complete={key(r) for r in accepted}==expected
    failed=len(current)-len(accepted)
    checks=dict(raw_hashes_and_prompt_contract=True,unique_slots=len(set(slots))==len(slots),
        only_planned_slots=set(slots)<=expected,global_denominator_unchanged=receipt['expected_predictions']==len(expected),
        success_count=receipt['successful_predictions']==len(accepted),failure_count=receipt['failed_predictions']==failed,
        pending_count=receipt['pending_predictions']==len(expected)-len(current),
        barrier_matches_actual_membership=receipt['prediction_barrier_closed']==complete,
        no_global_result_claim=receipt.get('global_result') is False)
    if not all(checks.values()): raise ValueError(checks)
    observations,scores=summarize_scores(accepted)
    # An unreturned draw is not counted as a measured unknown; distinguish both.
    coverage=[]
    for dimension,fields in FIELDS.items():
        for field in fields:
            eids={e['entity_id'] for e in entities if e['dimension']==dimension}
            obs=observations.loc[observations.field.eq(field)]
            grouped=obs.groupby('entity_id')
            full=sum(len(g)==len(MODELS)*REPEATS for _,g in grouped)
            coverage.append(dict(dimension=dimension,field=field,expected_entities=len(eids),
                full_response_entities=full,expected_draws=len(eids)*len(MODELS)*REPEATS,
                returned_draws=len(obs),numeric_draws=int(obs.score.notna().sum()),
                explicit_null_draws=int(obs.score.isna().sum()),
                low_confidence_draws=int(obs.confidence.eq('low').sum())))
    output=GLOBAL/'campaign_validation'/receipt_path.stem
    immutable_bytes(output/'coverage.csv',pd.DataFrame(coverage).to_csv(index=False,lineterminator='\n').encode())
    if len(scores):
        scores['required_observations_per_field']=len(MODELS)*REPEATS
        immutable_bytes(output/'available_entity_scores.csv',scores.to_csv(index=False,lineterminator='\n').encode())
    report=dict(status='all_response_slots_available_not_accuracy_validation' if complete else 'partial_coverage_final_evaluation_blocked',
        all_integrity_checks_passed=True,checks=checks,source_receipt=str(receipt_path.relative_to(ROOT)),
        source_receipt_sha256=sha(receipt_path),expected_predictions=len(expected),successful_predictions=len(accepted),
        failed_predictions=failed,unattempted_predictions=len(expected)-len(current),
        global_prediction_barrier_closed=complete,scientific_accuracy_validated=False,global_potential_computed=False,
        coverage=coverage,source_code_sha256=sha(Path(__file__)),
        note='Confidence is model self-report. Partial field averages are not final global coefficients.')
    freeze_json(output/'validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'coverage','checks'}},indent=2))
    if require_complete: require_full_membership(current,expected)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    validate(args.receipt,args.require_complete)
