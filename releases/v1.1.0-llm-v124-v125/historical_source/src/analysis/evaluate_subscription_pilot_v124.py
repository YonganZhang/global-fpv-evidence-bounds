"""Consume frozen real SDK predictions and test L2/L3 wiring, not accuracy.

No SDK imports or model calls here. A closed prediction barrier is required
before joining judgments to the actual inventory, hard gates, and costs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, OUT, FIELDS, build_entity, build_masks, compute_levels
from src.analysis.run_subscription_fpv_scores_v124 import validate_prediction


def load_predictions(run_id):
    report_path = OUT/'reports'/(run_id+'_subscription_calls.json')
    report = json.loads(report_path.read_text())
    if not report['prediction_barrier_closed'] or report['successful_calls'] != report['planned_calls']:
        raise ValueError('Predictions incomplete: evaluation is blocked')
    run = OUT/'raw_subscription'/run_id/'run.json'
    contract = json.loads(run.read_text())
    entities = {r['entity_id']:r for r in contract['entities']}
    expected = {(e,p,r) for e in entities for p in contract['providers'] for r in range(contract['repeats'])}
    records, hashes, seen = [], {str(report_path.relative_to(ROOT)):hashlib.sha256(report_path.read_bytes()).hexdigest(),
        str(run.relative_to(ROOT)):hashlib.sha256(run.read_bytes()).hexdigest()}, set()
    for ref in report['responses']:
        path = ROOT/ref['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Prediction changed after barrier')
        record = json.loads(path.read_text())
        entity = entities[record['entity_id']]
        if record.get('environment') != contract['environment'] or record.get('dimension') != entity['dimension'] or record.get('context') != entity['context']:
            raise ValueError('Prediction from a different input/runtime contract')
        validate_prediction(record, entities[record['entity_id']])
        key = record['entity_id'],record['provider'],record['repeat']
        if key in seen or record['status'] != 'ok':
            raise ValueError('Duplicate or failed prediction')
        seen.add(key); records.append(record); hashes[ref['path']] = ref['sha256']
    if seen != expected:
        raise ValueError('Prediction membership differs from plan')
    return records, contract, hashes


def validate_case_entities(cases, entities):
    lookup = {e['entity_id']:e for e in entities}
    for row in cases.to_dict('records'):
        contexts = {
            'ecology':dict(waterbody_type=row['wb_type'],terrestrial_ecoregion=row['ecoregion'],biome=row['biome']),
            'engineering':dict(waterbody_type=row['wb_type'],koppen_zone=row['koppen_zone'],country=row['country_key']),
            'country':dict(country=row['country_key'],assessment_date='2026-09-08'),
            'province':dict(country=row['country_key'],province=row['province'],assessment_date='2026-09-08'),
        }
        for dim,ctx in contexts.items():
            expected = build_entity(dim,ctx)
            if row[dim+'_entity_id'] != expected['entity_id'] or lookup.get(expected['entity_id']) != expected:
                raise ValueError('Wrong dimension/context mapped to waterbody')


def summarize_scores(records):
    rows = []
    for record in records:
        for field, item in record['scores'].items():
            rows.append(dict(entity_id=record['entity_id'], dimension=record['dimension'],
                provider=record['provider'],model=record['response']['model'],repeat=record['repeat'],
                field=field, score=item['score'],confidence=item['confidence'], reason=item['reason']))
    observations = pd.DataFrame(rows)
    summaries = []
    for (entity,dim,field), data in observations.groupby(['entity_id','dimension','field'],sort=False):
        provider_means = data.groupby('provider').score.mean().dropna()
        repeats = data.groupby('provider').score.agg(['count','min','max'])
        repeat_range = (repeats['max']-repeats['min']).where(repeats['count'].ge(2))
        summaries.append(dict(entity_id=entity,dimension=dim,field=field,
            score=float(provider_means.mean()) if len(provider_means) else None,
            valid_observations=int(data.score.notna().sum()), missing_observations=int(data.score.isna().sum()),
            valid_providers=len(provider_means), expected_observations=len(data),
            model_mean_range=float(provider_means.max()-provider_means.min()) if len(provider_means)>=2 else None,
            max_repeat_range=float(repeat_range.max()) if repeat_range.notna().any() else None,
            low_confidence_observations=int(data.confidence.eq('low').sum()),
            aggregation='mean within each provider, then equal mean across available providers'))
    return observations, pd.DataFrame(summaries)


def compute_case_results(cases, score_map, arm):
    masks = build_masks(cases)
    records = []
    def lookup(entity, field):
        value = score_map.get((entity,field))
        return float(value) if value is not None and pd.notna(value) else None
    for index,row in cases.iterrows():
        scores = dict(ecology=lookup(row.ecology_entity_id,'ecological_suitability'),
            maintenance=lookup(row.engineering_entity_id,'maintenance_ease'),
            support=lookup(row.country_entity_id,'renewable_support'),
            finance=lookup(row.country_entity_id,'finance_delivery'),
            province_offset=lookup(row.province_entity_id,'implementation_offset'))
        base = dict(l1=float(row.l0_type_specific_generation_gwh), cost=float(row.lcoe_fpv_reference_usd_mwh_v117))
        for endpoint in ['lower','upper']:
            gates = dict(env_pass=bool(masks['L2_'+endpoint].loc[index]),access_pass=bool(masks['access_'+endpoint].loc[index]))
            for strength in [0.,.25,.5,.75]:
                for missing_case in ['unadjusted','restricted']:
                    result = compute_levels(**base,**gates,**scores,strength=strength,missing_case=missing_case)
                    records.append(dict(wb_id=int(row.wb_id),country=row.country_key,waterbody_type=row.wb_type,
                        province=row.province,arm=arm,endpoint=endpoint,strength=strength,missing_case=missing_case,
                        base_cost_usd_mwh=base['cost'],**gates,**scores,**result))
    return pd.DataFrame(records)


def run_evaluation(run_id):
    records, contract, hashes = load_predictions(run_id)
    cases_path = OUT/'data/pilot_cases.parquet'
    cases = pd.read_parquet(cases_path)
    if not cases.wb_id.is_unique:
        raise ValueError('Duplicate case ID')
    validate_case_entities(cases,contract['entities'])
    observations,scores = summarize_scores(records)
    score_map = {(r.entity_id,r.field):r.score for r in scores.itertuples()}
    levels = [compute_case_results(cases,score_map,'equal_provider_mean')]
    for (provider,repeat), data in observations.groupby(['provider','repeat']):
        levels.append(compute_case_results(cases,{(r.entity_id,r.field):r.score for r in data.itertuples()},f'{provider}_repeat_{repeat}'))
    # Controls cannot select/tune coefficients; they only show mapping behavior.
    constant = {(r.entity_id,r.field):(0. if r.field=='implementation_offset' else .5) for r in scores.itertuples()}
    levels.append(compute_case_results(cases,constant,'uniform_0.5_control'))
    rng = np.random.default_rng(20260908)
    random_scores = {(r.entity_id,r.field):float(rng.uniform(*FIELDS[r.dimension][r.field])) for r in scores.itertuples()}
    levels.append(compute_case_results(cases,random_scores,'random_control_seed_20260908'))
    levels = pd.concat(levels,ignore_index=True)
    checks = dict(hierarchy=bool(((levels.L3_gwh<=levels.L2_gwh+1e-9)&(levels.L2_gwh<=levels.L1_gwh+1e-9)&levels.L3_gwh.ge(0)).all()),
        finite_outputs=bool(np.isfinite(levels[['L1_gwh','L2_gwh','L3_gwh','adjusted_cost_usd_mwh']]).all().all()),
        unique_results=not levels.duplicated(['wb_id','arm','endpoint','strength','missing_case']).any(),
        hard_exclusions=bool(levels.loc[~levels.env_pass,'L2_gwh'].eq(0).all() and levels.loc[~levels.access_pass,'L3_gwh'].eq(0).all()),
        cost_exclusions=bool(levels.loc[levels.adjusted_cost_usd_mwh.gt(100),'L3_gwh'].eq(0).all()))
    zero = levels.loc[levels.strength.eq(0)]
    checks['zero_strength_exact'] = bool((zero.L2_gwh==zero.L1_gwh*zero.env_pass).all()
        and (zero.L3_gwh==zero.L1_gwh*zero.env_pass*zero.access_pass*zero.base_cost_usd_mwh.le(100)).all())
    # Independently reconstruct the nonzero point from present-value components.
    central = levels.loc[levels.arm.eq('equal_provider_mean')&levels.strength.eq(.5)&levels.missing_case.eq('unadjusted')]
    om_pv = .02 * sum(1/1.08**t for t in range(1,26))
    reconstructed = []
    for row in central.itertuples():
        e,m,s,f = [v if pd.notna(v) else 1. for v in [row.ecology,row.maintenance,row.support,row.finance]]
        p = row.province_offset if pd.notna(row.province_offset) else 0.
        ecost = row.base_cost_usd_mwh*(1+(1+.5*(1-m))*om_pv)/(1+om_pv)
        l2 = row.L1_gwh*row.env_pass*(.5+.5*e)
        l3 = l2*row.access_pass*(ecost<=100)*(.5+.5*np.clip((s+f)/2+p,0,1))
        reconstructed.append(np.allclose([ecost,l2,l3],[row.adjusted_cost_usd_mwh,row.L2_gwh,row.L3_gwh],rtol=1e-12))
    checks['independent_present_value_reconstruction'] = all(reconstructed)
    # The deliberately eligible sample contains no actual environmental reject.
    # Exercise negative gates separately, labelled as constructed interventions.
    negative_checks = []
    for row in central.itertuples():
        s = {key:(float(getattr(row,key)) if pd.notna(getattr(row,key)) else None)
             for key in ['ecology','maintenance','support','finance','province_offset']}
        excluded = compute_levels(row.L1_gwh,False,True,row.base_cost_usd_mwh,**s)
        costly = compute_levels(row.L1_gwh,True,True,200.,**s)
        negative_checks.append(excluded['L2_gwh']==excluded['L3_gwh']==0 and costly['L3_gwh']==0)
    checks['constructed_negative_gate_interventions'] = all(negative_checks)
    # Ablate real model components individually; no new scores or thresholds.
    ablations = []
    for label, changes in [('no_ecology',{'ecological_suitability':1.}),
            ('no_maintenance',{'maintenance_ease':1.}),
            ('no_institutions',{'renewable_support':1.,'finance_delivery':1.,'implementation_offset':0.})]:
        altered = {(entity,field):changes.get(field,value) for (entity,field),value in score_map.items()}
        part = compute_case_results(cases,altered,label)
        ablations.append(part.loc[part.strength.eq(.5)&part.missing_case.eq('unadjusted')])
    ablations = pd.concat(ablations,ignore_index=True)
    coverage = observations.groupby('dimension',as_index=False).agg(requested_field_outputs=('score','size'),
        valid_field_outputs=('score','count'),low_confidence_outputs=('confidence',lambda x:int(x.eq('low').sum())))
    model_reasons_need_review = observations.loc[observations.confidence.isin(['low','unknown'])]
    directory = OUT/'reports'/run_id
    directory.mkdir(parents=True,exist_ok=True)
    for name,table in [('observations',observations),('entity_scores',scores),('case_levels',levels),
                       ('central_cases',central),('ablations',ablations),('coverage',coverage),
                       ('low_confidence_review',model_reasons_need_review)]:
        table.to_csv(directory/(name+'.csv'),index=False)
    hashes[str(cases_path.relative_to(ROOT))] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    for name in ['fpv_llm_v124.py','evaluate_subscription_pilot_v124.py']:
        path = ROOT/'src/analysis'/name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(status='real_predictions_consumed_pilot_only', all_wiring_checks_passed=all(checks.values()),
        checks=checks, waterbodies=len(cases), entities=len(contract['entities']),
        model_calls=len(records), providers=contract['providers'], repeats=contract['repeats'],
        valid_field_outputs=int(observations.score.notna().sum()), total_field_outputs=len(observations),
        fully_two_provider_fields=int(scores.valid_providers.eq(2).sum()), entity_fields=len(scores),
        observed_gate_coverage=dict(environment_excluded_case_endpoints=int((~central.env_pass).sum()),
            access_excluded_case_endpoints=int((~central.access_pass).sum()),
            cost_excluded_case_endpoints=int(central.adjusted_cost_usd_mwh.gt(100).sum())),
        environment_negative_coverage='not_exercised_by_real_sample; constructed_gate_interventions_only',
        L1_unchanged=True, L2_consumes_ecology=True, L3_consumes_maintenance_and_institutions=True,
        global_result=False, scientific_accuracy_validated=False,
        source_hashes=hashes, output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.csv')},
        limitations=['Four intentionally eligible cases are a wiring sample, not representative of the globe.',
            'Area/cost mappings remain researcher-defined sensitivity assumptions, not field-calibrated.',
            'Missing treatments are separate scenarios, not a statistical confidence interval.',
            'Repeat/model disagreement measures consistency, not correctness.',
            'Fixed contemporary judgments are not future governance predictions.'])
    (directory/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'source_hashes','output_hashes'}},indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    run_evaluation(parser.parse_args().run_id)
