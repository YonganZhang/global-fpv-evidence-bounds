"""Reduced-plan consumers: Codex repeat 0 everywhere; Claude is diagnostic only.

Every entry checks the reduced completeness barrier before reading numerical
inputs. Original physical, decomposition and benefit mathematics are reused;
the former equal-provider arm label is adapted explicitly, never averaged.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY, TAGS, FIELDS
from src.analysis.prepare_global_scores_v124 import GLOBAL, sha, freeze_json, immutable_bytes
from src.analysis.global_score_inputs_v124 import SCORE_FIELDS, SCORE_COLUMNS, load_catalogue
from src.analysis.resume_global_scores_v124 import read_refs
from src.analysis.evaluate_global_levels_v124 import (
    evaluate_partition as original_evaluate_partition, check_baseline,
    physical_partitions, load_sources, GROUP_FIELDS, ICE_CASES)
from src.analysis.analyze_global_future_v124 import (
    decompose_pair as original_decompose_pair, read_weighted, read_ice, CONFIG, KEYS)
from src.analysis.global_benefit_metrics_v124 import (
    present_benefits as original_present_benefits, demand_by_iso)
from src.analysis.analyze_global_benefits_v124 import EVAPORATION, DEMAND, ISO
from src.future.align_llm_scores import country_key

LEAN = GLOBAL / 'lean'
PRIMARY_ARM = 'codex_primary'
CURRENT = ('current_reference', 'current', 'POWER_2019_2023', 'current_inventory')
CODE_FILES = [
    'src/analysis/lean_global_consumers_v124.py',
    'src/analysis/lean_global_plan_v124.py',
    'src/analysis/run_lean_global_scores_v124.py',
    'src/analysis/parallel_lean_dispatch_v124.py',
    'src/analysis/recover_lean_scores_v124.py',
    'src/analysis/global_score_inputs_v124.py',
    'src/analysis/evaluate_global_levels_v124.py',
    'src/analysis/global_level_kernel_v124.py',
    'src/analysis/analyze_global_future_v124.py',
    'src/analysis/global_benefit_metrics_v124.py',
    'src/analysis/analyze_global_benefits_v124.py',
    'src/analysis/resume_global_scores_v124.py',
    'src/analysis/run_global_subscription_v124.py',
    'src/analysis/run_subscription_fpv_scores_v124.py',
    'src/analysis/fpv_subscription_worker.py',
    'src/analysis/run_direct_fpv_scores_v124.py',
    'src/analysis/validate_global_campaign_v124.py',
    'src/analysis/fpv_llm_v124.py',
    'src/analysis/prepare_global_scores_v124.py',
    'src/future/align_llm_scores.py',
]


def validate_receipt(receipt_path, require_complete=False):
    # Lazy imports keep the pure numerical functions independent of the runner.
    # This also checks dispatch history for legacy receipts; never bypass it
    # based on whether the requested receipt itself has dispatch metadata.
    path = (ROOT / receipt_path).resolve()
    if path.is_relative_to((LEAN / 'recovery').resolve()):
        from src.analysis.recover_lean_scores_v124 import validate_receipt as validate
    else:
        from src.analysis.parallel_lean_dispatch_v124 import validate_receipt as validate
    return validate(receipt_path, require_complete=require_complete)


def load_plan():
    from src.analysis.lean_global_plan_v124 import load_plan as load
    return load()


def code_hashes():
    return {p: sha(ROOT / p) for p in CODE_FILES}


def write_table(path, table):
    if path.suffix == '.parquet':
        payload = BytesIO()
        table.to_parquet(payload, index=False)
        immutable_bytes(path, payload.getvalue())
    else:
        immutable_bytes(path, table.to_csv(index=False, lineterminator='\n').encode())


def assemble_primary_inputs(mapping, entities, records, waterbody_ids, required_slots):
    """Join raw primary scores without audit fallback, averaging or imputation."""
    lookup = {e['entity_id']: e for e in entities}
    if not entities or len(lookup) != len(entities) or any(e['dimension'] not in FIELDS for e in entities):
        raise ValueError('Invalid entity catalogue')
    expected = {(s['entity_id'], s['provider'], s['repeat']) for s in required_slots}
    if len(expected) != len(required_slots):
        raise ValueError('Duplicated required slots')
    primary_slots = {(e, 'codex', 0) for e in lookup}
    if {s for s in expected if s[1] == 'codex'} != primary_slots:
        raise ValueError('Primary slots must cover exactly the catalogue')
    if any(e not in lookup or p not in {'codex', 'claude'} or type(r) is not int or r != 0
           for e, p, r in expected):
        raise ValueError('Noncanonical required slot')
    observed, observations = {}, []
    for record in records:
        slot = (record['entity_id'], record['provider'], record['repeat'])
        if (type(record['repeat']) is not int or record['repeat'] != 0 or slot in observed
            or slot not in expected or record.get('status') != 'ok'):
            raise ValueError('Duplicate, extra or unsuccessful required record')
        entity = lookup[record['entity_id']]
        if record['dimension'] != entity['dimension'] or set(record['scores']) != set(FIELDS[entity['dimension']]):
            raise ValueError('Wrong record dimension or fields')
        observed[slot] = record
        for field, item in record['scores'].items():
            value = item['score']
            lo, hi = FIELDS[entity['dimension']][field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                     or not np.isfinite(value) or not lo <= value <= hi):
                raise ValueError('Invalid primary or audit score')
            observations.append(dict(entity_id=slot[0], dimension=entity['dimension'],
                provider=slot[1], repeat=0, role='primary' if slot[1] == 'codex' else 'audit',
                field=field, score=value, confidence=item.get('confidence'), reason=item.get('reason')))
    if set(observed) != expected:
        raise ValueError('Incomplete required raw records')
    obs = pd.DataFrame(observations)
    primary = obs.loc[obs.provider.eq('codex')].copy()
    primary['numeric_draws'] = primary.score.notna().astype(int)
    primary['null_draws'] = primary.score.isna().astype(int)
    primary['numeric_providers'] = primary.numeric_draws
    ids = pd.Index(waterbody_ids, name='wb_id')
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError('Invalid inventory IDs')
    if mapping.wb_id.isna().any() or not mapping.wb_id.is_unique or set(mapping.wb_id) != set(ids):
        raise ValueError('Mapping does not cover the exact inventory')
    joined = mapping.set_index('wb_id').loc[ids].reset_index()
    for dimension, field in SCORE_FIELDS:
        column = dimension + '_entity_id'
        valid_ids = {e['entity_id'] for e in entities if e['dimension'] == dimension}
        if not set(joined[column].dropna()) <= valid_ids:
            raise ValueError('Unknown or cross-dimension mapped entity')
        table = primary.loc[primary.field.eq(field)].set_index('entity_id')
        joined[field] = joined[column].map(table.score).astype(float)
        joined[field + '_context_missing'] = joined[column].isna()
        for count in ['numeric_draws', 'null_draws', 'numeric_providers']:
            joined[field + '_' + count] = joined[column].map(table[count]).fillna(0).astype(int)
    audit = obs.loc[obs.provider.eq('claude'), ['entity_id', 'dimension', 'field', 'score']].rename(columns={'score': 'claude_score'})
    pairs = audit.merge(primary[['entity_id', 'dimension', 'field', 'score']].rename(columns={'score': 'codex_score'}),
        on=['entity_id', 'dimension', 'field'], how='left', validate='one_to_one')
    pairs['codex_is_null'] = pairs.codex_score.isna()
    pairs['claude_is_null'] = pairs.claude_score.isna()
    pairs['null_disagreement'] = pairs.codex_is_null != pairs.claude_is_null
    pairs['claude_minus_codex'] = pairs.claude_score - pairs.codex_score
    pairs['absolute_delta'] = pairs.claude_minus_codex.abs()
    return joined, obs, primary, pairs


def build_inputs(receipt_path):
    receipt_path = (ROOT / receipt_path).resolve()
    report = validate_receipt(receipt_path, require_complete=True)
    if not report['all_integrity_checks_passed'] or not report['prediction_barrier_closed']:
        raise ValueError('Reduced prediction barrier is incomplete')
    plan = load_plan()
    receipt = json.loads(receipt_path.read_text())
    contract = json.loads((receipt_path.parent.parent / 'contract.json').read_text())
    preparation_path = GLOBAL / 'preparation.json'
    preparation = json.loads(preparation_path.read_text())
    if not preparation['all_checks_passed'] or preparation['waterbodies'] != 199976 or preparation['entities'] != 4701:
        raise ValueError('Unexpected global preparation')
    for path in [INVENTORY, TAGS]:
        if sha(path) != preparation['source_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('Inventory or context changed')
    for name in ['entities.json', 'waterbody_entity_map.csv']:
        if sha(GLOBAL / name) != preparation['artifact_hashes'][name]:
            raise ValueError('Frozen mapping or catalogue changed')
    entities = load_catalogue(GLOBAL / 'entities.json')
    records = list(read_refs(receipt['responses'], {e['entity_id']: e for e in entities}, contract['environment']).values())
    mapping = pd.read_csv(GLOBAL / 'waterbody_entity_map.csv')
    ids = pd.read_parquet(INVENTORY, columns=['wb_id']).wb_id
    if len(ids) != 199976 or len(entities) != 4701:
        raise ValueError('Actual inventory/catalogue size differs from design')
    joined, observations, primary, pairs = assemble_primary_inputs(mapping, entities, records, ids, plan['required_slots'])
    audit_strata = {entity_id: stratum for stratum in plan['strata'] for entity_id in stratum['selected_ids']}
    for name in ['region', 'waterbody_type', 'population', 'sample_size']:
        pairs['stratum_' + name] = pairs.entity_id.map(lambda entity_id: audit_strata[entity_id][name])
    destination = LEAN / 'score_inputs' / receipt_path.stem
    tables = {'waterbody_scores.parquet': joined, 'observations.csv': observations,
              'entity_scores.csv': primary, 'audit_pairs.csv': pairs}
    for name, table in tables.items():
        write_table(destination / name, table)
    result = dict(status='complete_reduced_score_inputs_not_potential_results', waterbodies=len(joined),
        entities=len(entities), expected_predictions=plan['expected_predictions'],
        primary_count=plan['primary_count'], audit_count=plan['audit_count'],
        prediction_barrier_closed=True, numerical_levels_complete=False, future_changes_complete=False,
        present_benefits_complete=False,
        source_receipt=str(receipt_path.relative_to(ROOT)), source_receipt_sha256=sha(receipt_path),
        plan_sha256=sha(LEAN / 'plan.json'), preparation_sha256=sha(preparation_path),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
            [INVENTORY, TAGS, GLOBAL / 'entities.json', GLOBAL / 'waterbody_entity_map.csv', preparation_path,
             LEAN / 'plan.json', receipt_path, receipt_path.parent.parent / 'contract.json']},
        code_hashes=code_hashes(), primary_arm=PRIMARY_ARM,
        aggregation='Codex canonical repeat 0 for every entity; Claude audit never changes primary weights',
        audit_interpretation='Unweighted sample pairs with strata; not population prevalence or scientific accuracy',
        score_column_order=list(SCORE_COLUMNS), defaults_filled=False, global_potential_computed=False,
        scientific_accuracy_validated=False, output_hashes={name: sha(destination / name) for name in tables})
    freeze_json(destination / 'validation.json', result)
    return result


def evaluate_partition(inventory, physical, scores):
    tables = original_evaluate_partition(inventory, physical, scores)
    return tuple(table.assign(arm=table.arm.replace({'equal_provider_mean': PRIMARY_ARM})) for table in tables)


def _mathematical_adapter(table):
    """Only translate a verified semantic label on a local, non-mutating copy."""
    if table.empty or not table.arm.eq(PRIMARY_ARM).all() or not table.strength.eq(.5).all():
        raise ValueError('Expected codex_primary strength-0.5 scenarios')
    return table.assign(arm='equal_provider_mean')


def decompose_pair(historical, future, historical_ice, future_ice):
    return original_decompose_pair(_mathematical_adapter(historical), _mathematical_adapter(future),
                                   historical_ice, future_ice)


def present_benefits(weighted, inventory, evaporation, demand):
    return original_present_benefits(_mathematical_adapter(weighted), inventory, evaporation, demand)


def _run_directory(run_id):
    if not run_id.isalnum():
        raise ValueError('Use an alphanumeric output run ID')
    return LEAN / 'weighted_levels' / run_id


def _expected_partitions(manifest):
    expected = {(r['model'], r['experiment'], '_'.join(map(str, r['period'])), ice)
                for r in manifest['results'] for ice in ICE_CASES}
    expected.add(CURRENT)
    return expected


def run_levels(receipt, run_id):
    destination = _run_directory(run_id)
    score_manifest = build_inputs(receipt)
    score_dir = LEAN / 'score_inputs' / Path(receipt).stem
    score_path = score_dir / 'waterbody_scores.parquet'
    if sha(score_path) != score_manifest['output_hashes'][score_path.name]:
        raise ValueError('Score input changed')
    manifest, baseline, sources = load_sources()
    sources.update(score_manifest['source_hashes'])
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in [score_dir / 'validation.json', score_path]})
    inventory = pd.read_parquet(INVENTORY)
    if len(inventory) != 199976:
        raise ValueError('Incomplete inventory')
    inventory['country_key'] = inventory.country_v117.fillna('').map(country_key)
    if inventory.country_key.eq('').any():
        raise ValueError('Unknown country aggregation membership')
    scores = pd.read_parquet(score_path)
    contract = dict(source_hashes=sources, code_hashes=code_hashes(),
        source_receipt=score_manifest['source_receipt'], source_receipt_sha256=score_manifest['source_receipt_sha256'],
        plan_sha256=score_manifest['plan_sha256'], primary_arm=PRIMARY_ARM,
        aggregation=score_manifest['aggregation'], detail_strength=.5, sensitivity_strengths=[0., .25, .5, .75],
        ablations=['no_ecology', 'no_maintenance', 'no_institutions'],
        assumptions='Fixed contemporary judgments; uncalibrated area/cost mappings; no future governance prediction')
    freeze_json(destination / 'contract.json', contract)
    global_tables, country_tables, outputs = [], [], []
    for group, physical in physical_partitions(inventory, manifest):
        details, countries, totals = evaluate_partition(inventory, physical, scores)
        check_baseline(totals, baseline, group)
        for field, value in group.items():
            countries[field], totals[field] = value, value
        path = destination / 'waterbody_partitions' / ('__'.join(group[f] for f in GROUP_FIELDS) + '.parquet')
        write_table(path, details)
        outputs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(details), **group))
        global_tables.append(totals)
        country_tables.append(countries)
        print('evaluated', path.stem, flush=True)
    if len(outputs) != 46 or {tuple(r[k] for k in GROUP_FIELDS) for r in outputs} != _expected_partitions(manifest):
        raise ValueError('Incomplete current plus future partition set')
    for name, tables in [('global_summary', global_tables), ('country_summary', country_tables)]:
        write_table(destination / (name + '.csv'), pd.concat(tables, ignore_index=True))
    for path, digest in {**sources, **contract['code_hashes']}.items():
        if sha(ROOT / path) != digest:
            raise ValueError('Source/code changed during levels computation: ' + path)
    report = dict(status='conditional_codex_primary_levels_pending_scientific_review',
        all_numerical_checks_passed=True, primary_arm=PRIMARY_ARM, waterbodies=len(inventory), physical_partitions=46,
        prediction_barrier_closed=True, numerical_levels_complete=True, future_changes_complete=False,
        present_benefits_complete=False,
        global_scenario_rows=sum(len(t) for t in global_tables),
        checks=['per_waterbody_hierarchy', 'unchanged_physical_L1', 'hard_exclusions', 'cost_gate',
                'zero_strength_physical_identity', 'country_global_sums', 'validated_zero_strength_baseline'],
        contract_sha256=sha(destination / 'contract.json'), waterbody_outputs=outputs,
        summary_hashes={name: sha(destination / name) for name in ['global_summary.csv', 'country_summary.csv']},
        scientific_accuracy_validated=False, paper_updated=False,
        note='Conditional scenario outputs; audit diagnostics are not weights or scientific accuracy')
    freeze_json(destination / 'validation.json', report)
    return report


def load_weighted_run(run_id):
    """Validate lean parent without invoking the legacy 18,804-draw barrier."""
    directory = _run_directory(run_id)
    contract_path = directory / 'contract.json'
    contract = json.loads(contract_path.read_text())
    report = json.loads((directory / 'validation.json').read_text())
    validated = validate_receipt(contract['source_receipt'], require_complete=True)
    if not validated['all_integrity_checks_passed'] or not validated['prediction_barrier_closed']:
        raise ValueError('Incomplete reduced parent receipt')
    if (not report['all_numerical_checks_passed'] or report['waterbodies'] != 199976
        or report['physical_partitions'] != 46 or report['contract_sha256'] != sha(contract_path)
        or contract['primary_arm'] != PRIMARY_ARM or report['primary_arm'] != PRIMARY_ARM
        or contract['plan_sha256'] != sha(LEAN / 'plan.json')):
        raise ValueError('Incomplete, changed or wrong-arm weighted run')
    if sha(ROOT / contract['source_receipt']) != contract['source_receipt_sha256']:
        raise ValueError('Source scoring receipt changed')
    if contract['code_hashes'] != code_hashes():
        raise ValueError('Weighted code provenance differs')
    for path, digest in contract['source_hashes'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Weighted run source changed: ' + path)
    for name, digest in report['summary_hashes'].items():
        if name not in {'global_summary.csv', 'country_summary.csv'} or sha(directory / name) != digest:
            raise ValueError('Weighted summary changed')
    if set(report['summary_hashes']) != {'global_summary.csv', 'country_summary.csv'}:
        raise ValueError('Incomplete summary hashes')
    manifest, _, _ = load_sources()
    lookup = {tuple(ref[k] for k in GROUP_FIELDS): ref for ref in report['waterbody_outputs']}
    if len(lookup) != 46 or len(report['waterbody_outputs']) != 46 or set(lookup) != _expected_partitions(manifest):
        raise ValueError('Weighted partition membership differs from physical design')
    # All 46 artifacts are verified, including partitions not read by benefits.
    for ref in lookup.values():
        path = (ROOT / ref['path']).resolve()
        if (not path.is_relative_to(directory.resolve() / 'waterbody_partitions')
            or sha(path) != ref['sha256'] or ref['rows'] != 199976 * 8):
            raise ValueError('Changed, misplaced or incomplete weighted partition')
    return directory, report, manifest, lookup


def run_future(run_id):
    directory, _, manifest, lookup = load_weighted_run(run_id)
    ids = pd.read_parquet(INVENTORY, columns=['wb_id']).wb_id
    sources = {(r['model'], r['experiment'], '_'.join(map(str, r['period']))): r for r in manifest['results']}
    destination = directory / 'future_changes'
    contract = dict(parent_validation_sha256=sha(directory / 'validation.json'), code_hashes=code_hashes(),
        primary_arm=PRIMARY_ARM, comparison='same climate model and ice error case against 1995_2014',
        ordered_decomposition='generation at historical admission, then admission at future generation',
        unknown_ice='separate subset, not an extra additive component or proof of thaw')
    freeze_json(destination / 'contract.json', contract)
    globals_, countries_, outputs = [], [], []
    cached_key, history, ice = None, None, None
    for key in sorted(lookup):
        model, experiment, period, ice_case = key
        if experiment in {'current', 'historical'}:
            continue
        historical_key = (model, 'historical', '1995_2014', ice_case)
        if cached_key != historical_key:
            history = read_weighted(lookup[historical_key], directory, ids)
            ice = read_ice(sources[historical_key[:3]], ice_case)
            cached_key = historical_key
        future = read_weighted(lookup[key], directory, ids)
        change = decompose_pair(history, future, ice, read_ice(sources[key[:3]], ice_case))
        group = dict(zip(GROUP_FIELDS, key))
        measures = [c for c in change if c not in KEYS + ['country']]
        countries = change.groupby(CONFIG + ['country'], sort=True)[measures].sum().reset_index()
        totals = change.groupby(CONFIG, sort=True)[measures].sum().reset_index()
        if not np.allclose(countries.groupby(CONFIG)[measures].sum().sort_index().to_numpy(float),
                           totals.set_index(CONFIG).sort_index()[measures].to_numpy(float), rtol=1e-10, atol=1e-6):
            raise ValueError('Country/global changes do not reconcile')
        for column, value in group.items():
            countries[column], totals[column] = value, value
        path = destination / 'waterbodies' / ('__'.join(key) + '.parquet')
        write_table(path, change)
        outputs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(change), **group))
        countries_.append(countries)
        globals_.append(totals)
        print('decomposed', path.stem, flush=True)
    if len(outputs) != 36:
        raise ValueError('Incomplete future comparison set')
    for name, tables in [('country_changes', countries_), ('global_changes', globals_)]:
        write_table(destination / (name + '.csv'), pd.concat(tables, ignore_index=True))
    if sha(directory / 'validation.json') != contract['parent_validation_sha256']:
        raise ValueError('Weighted parent changed during future computation')
    result = dict(status='conditional_future_changes_not_future_governance', all_checks_passed=True,
        primary_arm=PRIMARY_ARM, waterbodies=199976, paired_partitions=36, outputs=outputs,
        prediction_barrier_closed=True, numerical_levels_complete=True, future_changes_complete=True,
        contract_sha256=sha(destination / 'contract.json'),
        summary_hashes={name: sha(destination / name) for name in ['country_changes.csv', 'global_changes.csv']},
        unknown_ice_separated=True, fixed_factors_checked=True, scientific_accuracy_validated=False)
    freeze_json(destination / 'validation.json', result)
    return result


def run_benefits(run_id):
    directory, _, _, lookup = load_weighted_run(run_id)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [INVENTORY, EVAPORATION, DEMAND, ISO]}
    parent_sha = sha(directory / 'validation.json')
    inventory = pd.read_parquet(INVENTORY)
    weighted = read_weighted(lookup[CURRENT], directory, inventory.wb_id)
    evaporation = pd.read_csv(EVAPORATION, usecols=['wb_id', 'annual_evap_mm'])
    demand = demand_by_iso(pd.read_csv(DEMAND), pd.read_csv(ISO))
    water, country, total = present_benefits(weighted, inventory, evaporation, demand)
    checks = dict(waterbody_rows=len(water) == 199976 * 8, scenario_totals=len(total) == 8)
    for level in ['L1', 'L2', 'L3']:
        original = weighted.groupby(CONFIG)[level + '_gwh'].sum().sort_index() / 1000.
        result = total.set_index(CONFIG)[level + '_generation_twh'].sort_index()
        checks[level + '_global_energy_reconciles'] = bool(np.allclose(original, result, rtol=1e-10, atol=1e-8))
    if not all(checks.values()):
        raise ValueError(checks)
    for path, digest in sources.items():
        if sha(ROOT / path) != digest:
            raise ValueError('Benefit source changed during read')
    if sha(directory / 'validation.json') != parent_sha:
        raise ValueError('Weighted parent changed during benefit computation')
    destination = directory / 'present_benefits'
    freeze_json(destination / 'contract.json', dict(parent_validation_sha256=parent_sha,
        source_hashes=sources, code_hashes=code_hashes(), primary_arm=PRIMARY_ARM,
        demand_year=2023, comparison='annual generation versus matched national final electricity consumption',
        evaporation='legacy historical Penman depth, coverage-limited, not future hydrology',
        reduction_on_covered_area=.46, same_level_area=True,
        missing_evaporation='NaN; global savings restricted to known subset'))
    tables = {'waterbody_benefits.parquet': water, 'country_benefits.csv': country, 'global_benefits.csv': total}
    for name, table in tables.items():
        write_table(destination / name, table)
    result = dict(status='conditional_present_benefits_not_realized_supply_or_future_hydrology',
        all_checks_passed=True, primary_arm=PRIMARY_ARM, checks=checks, waterbodies=199976, scenario_totals=8,
        prediction_barrier_closed=True, numerical_levels_complete=True, present_benefits_complete=True,
        evaporation_known_waterbodies=int(water.loc[water.evaporation_known, 'wb_id'].nunique()),
        demand_source_countries=len(demand), contract_sha256=sha(destination / 'contract.json'),
        output_hashes={name: sha(destination / name) for name in tables}, scientific_accuracy_validated=False,
        note='46% is an assumed evaporation reduction. Annual demand ratios do not imply reliable self-sufficiency.')
    freeze_json(destination / 'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['inputs', 'levels', 'future', 'benefits']:
        command = commands.add_parser(name)
        if name in {'inputs', 'levels'}:
            command.add_argument('--receipt', required=True)
        if name != 'inputs':
            command.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.command == 'inputs':
        result = build_inputs(args.receipt)
    elif args.command == 'levels':
        result = run_levels(args.receipt, args.run_id)
    elif args.command == 'future':
        result = run_future(args.run_id)
    else:
        result = run_benefits(args.run_id)
    print(json.dumps({k: v for k, v in result.items() if k not in {'outputs', 'waterbody_outputs'}}, indent=2))


if __name__ == '__main__':
    main()
