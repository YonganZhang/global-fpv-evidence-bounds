"""Full current/future conditional levels after complete subscription scoring.

Reuses the frozen numerical kernel. Main strength 0.5 has per-waterbody output;
strength sensitivity has country/global totals; component ablations have global
totals. No observed-policy, score-accuracy or future-governance claim is made.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY
from src.analysis.prepare_global_scores_v124 import GLOBAL, freeze_json, immutable_bytes, sha
from src.analysis.global_score_inputs_v124 import build_inputs, SCORE_COLUMNS
from src.analysis.global_level_kernel_v124 import masks, levels
from src.future.align_llm_scores import country_key

ENERGY = ['L1_gwh', 'L2_gwh', 'L3_gwh']
GROUP_FIELDS = ['model', 'experiment', 'period', 'ice_error_case']
ICE_CASES = {'central', 'historical_error_shorter_ice', 'historical_error_longer_ice'}


def evaluate_partition(inventory, physical, score_table):
    """Synthetic-testable numerical path, without writing or fetching scores."""
    for frame in [inventory, physical, score_table]:
        if frame.empty or frame.wb_id.isna().any() or not frame.wb_id.is_unique:
            raise ValueError('Duplicate/missing waterbody ID')
    ids = physical.wb_id
    if set(inventory.wb_id) != set(ids) or set(score_table.wb_id) != set(ids):
        raise ValueError('Waterbody membership differs across physical/gate/score inputs')
    frame = inventory.set_index('wb_id', drop=False).loc[ids].reset_index(drop=True)
    scores = score_table.set_index('wb_id').loc[ids, list(SCORE_COLUMNS)].to_numpy(float)
    physical = physical.reset_index(drop=True)
    if frame.country_key.isna().any():
        raise ValueError('Missing aggregation country key')
    ice = physical.ice_duration_days_used.to_numpy() if 'ice_duration_days_used' in physical else None
    variants = [('equal_provider_mean', scores, [0., .25, .5, .75])]
    for name, columns, defaults in [('no_ecology', [0], [1.]), ('no_maintenance', [1], [1.]),
                                    ('no_institutions', [2, 3, 4], [1., 1., 0.])]:
        ablated = scores.copy()
        ablated[:, columns] = defaults
        variants.append((name, ablated, [.5]))
    detail, countries, global_rows = [], [], []
    for protection in ['core', 'conservative']:
        for endpoint in ['lower', 'upper']:
            env, access = masks(frame, endpoint, protection, ice)
            for arm, values, strengths in variants:
                for strength in strengths:
                    for missing_case in ['unadjusted', 'restricted']:
                        config = dict(protection=protection, endpoint=endpoint, arm=arm,
                                      strength=strength, missing_case=missing_case)
                        result = levels(physical.L1_gwh, physical.lcoe_usd_mwh, env, access, values,
                                        strength, missing_case)
                        valid = (np.isfinite(result.to_numpy()).all()
                            and (result.L3_gwh >= 0).all()
                            and (result.L3_gwh <= result.L2_gwh + 1e-9).all()
                            and (result.L2_gwh <= result.L1_gwh + 1e-9).all()
                            and np.array_equal(result.L1_gwh.to_numpy(), physical.L1_gwh.to_numpy())
                            and result.loc[~env, 'L2_gwh'].eq(0).all()
                            and result.loc[~access, 'L3_gwh'].eq(0).all()
                            and result.loc[result.adjusted_cost_usd_mwh.gt(100), 'L3_gwh'].eq(0).all())
                        if not valid:
                            raise ValueError('Hierarchy, physical identity, gate or cost invariant failed')
                        if strength == 0:
                            expected_l2 = physical.L1_gwh.to_numpy() * env
                            expected_l3 = expected_l2 * access * physical.lcoe_usd_mwh.le(100).to_numpy()
                            if not (np.array_equal(result.L2_gwh, expected_l2) and np.array_equal(result.L3_gwh, expected_l3)):
                                raise ValueError('Zero-strength control differs from physical gates')
                        sums = result[ENERGY].sum().to_numpy() / 1000
                        global_rows.append(dict(**config, waterbodies=len(frame),
                            **dict(zip(['L1_TWh', 'L2_TWh', 'L3_TWh'], sums)),
                            L2_fraction_L1=float(sums[1] / sums[0]) if sums[0] else None,
                            L3_fraction_L1=float(sums[2] / sums[0]) if sums[0] else None,
                            environmental_candidates=int(env.sum()), L3_positive_waterbodies=int(result.L3_gwh.gt(0).sum()),
                            cost_excluded_after_access=int((env & access & result.adjusted_cost_usd_mwh.gt(100)).sum())))
                        if arm == 'equal_provider_mean':
                            country = result[ENERGY].assign(country=frame.country_key.to_numpy()).groupby('country', sort=True).sum() / 1000
                            country.columns = ['L1_TWh', 'L2_TWh', 'L3_TWh']
                            if not np.allclose(country.sum().to_numpy(), sums, rtol=1e-12, atol=1e-10):
                                raise ValueError('Country and global sums differ')
                            countries.append(country.reset_index().assign(**config))
                            if strength == .5:
                                detail.append(result.assign(wb_id=ids.to_numpy(), country=frame.country_key.to_numpy(),
                                    base_cost_usd_mwh=physical.lcoe_usd_mwh.to_numpy(),
                                    env_pass=env, access_pass=access, **config))
    return pd.concat(detail, ignore_index=True), pd.concat(countries, ignore_index=True), pd.DataFrame(global_rows)


def load_sources():
    """Require the same 15 verified climate partitions as the no-LLM baseline."""
    manifest_path = ROOT / '_outputs/v122/reports/global_computation.json'
    validation_path = ROOT / '_outputs/v122/reports/global_validation.json'
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    if (validation['status'] != 'passes_declared_conditional_global_checks'
        or validation['source_computation_manifest_sha256'] != sha(manifest_path)
        or not all(all(c.values()) for c in validation['partition_checks'].values())
        or manifest['inputs'][str(INVENTORY.relative_to(ROOT))] != sha(INVENTORY)):
        raise ValueError('Future inputs are not verified for this inventory')
    accepted = {r['path']: r['sha256'] for r in validation['results']}
    expected = {(m, e, p) for m in ['MPI-ESM1-2-LR', 'MRI-ESM2-0', 'UKESM1-0-LL'] for e, p in [
        ('historical', (1995, 2014)), ('ssp126', (2031, 2050)), ('ssp126', (2081, 2100)),
        ('ssp585', (2031, 2050)), ('ssp585', (2081, 2100))]}
    if (len(manifest['results']) != 15 or len(accepted) != 15
        or accepted != {r['path']: r['sha256'] for r in manifest['results']}
        or {(r['model'], r['experiment'], tuple(r['period'])) for r in manifest['results']} != expected):
        raise ValueError('Missing, duplicated or changed future combinations')
    baseline_path = GLOBAL / 'no_llm_baseline_v2/validation.json'
    baseline = json.loads(baseline_path.read_text())
    baseline_table_path = baseline_path.parent / 'level_summary.csv'
    if (not baseline['all_checks_passed'] or not all(baseline['checks'].values())
        or baseline['llm_scores_applied'] is not False or baseline['output_sha256'] != sha(baseline_table_path)
        or baseline['source_hashes'][str(INVENTORY.relative_to(ROOT))] != sha(INVENTORY)
        or baseline['source_hashes'][str(manifest_path.relative_to(ROOT))] != sha(manifest_path)
        or baseline['code_hashes']['src/analysis/global_level_kernel_v124.py'] != sha(ROOT / 'src/analysis/global_level_kernel_v124.py')):
        raise ValueError('No-LLM baseline changed or not verified')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [INVENTORY, manifest_path, validation_path, baseline_path, baseline_table_path]}
    sources.update(accepted)
    return manifest, pd.read_csv(baseline_table_path), sources


def physical_partitions(inventory, manifest):
    yield dict(model='current_reference', experiment='current', period='POWER_2019_2023', ice_error_case='current_inventory'), pd.DataFrame(dict(
        wb_id=inventory.wb_id, L1_gwh=inventory.l0_type_specific_generation_gwh,
        lcoe_usd_mwh=inventory.lcoe_fpv_reference_usd_mwh_v117))
    for source in manifest['results']:
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('Future physical source changed')
        data = pd.read_parquet(path, columns=['wb_id', 'L1_gwh', 'lcoe_usd_mwh', 'ice_duration_days_used', 'ice_error_case'])
        if len(data) != len(inventory) * 3 or set(data.ice_error_case) != ICE_CASES:
            raise ValueError('Future ice variants differ from frozen design')
        for ice_case, sub in data.groupby('ice_error_case', sort=True):
            yield dict(model=source['model'], experiment=source['experiment'], period='_'.join(map(str, source['period'])),
                       ice_error_case=ice_case), sub.reset_index(drop=True)


def check_baseline(summary, baseline, group):
    selection = np.ones(len(baseline), dtype=bool)
    for field, value in group.items():
        selection &= baseline[field].eq(value).to_numpy()
    reference = baseline.loc[selection].set_index(['protection', 'endpoint'])
    if len(reference) != 4 or not reference.index.is_unique:
        raise ValueError('Missing or duplicated no-LLM baseline rows')
    zero = summary.loc[summary.strength.eq(0)]
    if len(zero) != 8 or zero.duplicated(['protection', 'endpoint', 'missing_case']).any():
        raise ValueError('Zero-strength scenarios incomplete or duplicated')
    for row in zero.itertuples():
        old = reference.loc[(row.protection, row.endpoint)]
        if not np.allclose([row.L1_TWh, row.L2_TWh, row.L3_TWh], old[['L1_TWh', 'L2_TWh', 'L3_TWh']].to_numpy(float), rtol=1e-11, atol=1e-9):
            raise ValueError('Zero-strength partition differs from validated no-LLM baseline')


def run(receipt, run_id):
    if not run_id.isalnum():
        raise ValueError('Use an alphanumeric output run ID')
    score_manifest = build_inputs(receipt)  # Real completeness gate, before numerical work.
    score_dir = GLOBAL / 'score_inputs' / Path(receipt).stem
    score_path = score_dir / 'waterbody_scores.parquet'
    if sha(score_path) != score_manifest['output_hashes'][score_path.name]:
        raise ValueError('Score input changed')
    manifest, baseline, sources = load_sources()
    sources[str(score_dir.relative_to(ROOT) / 'validation.json')] = sha(score_dir / 'validation.json')
    sources[str(score_path.relative_to(ROOT))] = sha(score_path)
    inventory = pd.read_parquet(INVENTORY)
    if len(inventory) != 199976:
        raise ValueError('Incomplete inventory')
    inventory['country_key'] = inventory.country_v117.fillna('').map(country_key)
    if inventory.country_key.eq('').any():
        raise ValueError('Unknown country aggregation membership')
    score_table = pd.read_parquet(score_path)
    destination = GLOBAL / 'weighted_levels' / run_id
    contract = dict(source_hashes=sources, source_receipt=score_manifest['source_receipt'],
        source_receipt_sha256=score_manifest['source_receipt_sha256'],
        code_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / 'src/analysis/global_level_kernel_v124.py', ROOT / 'src/analysis/global_score_inputs_v124.py']},
        detail_strength=.5, sensitivity_strengths=[0., .25, .5, .75],
        ablations=['no_ecology', 'no_maintenance', 'no_institutions'],
        assumptions='Fixed contemporary judgments; uncalibrated area/cost mappings; no future governance prediction.')
    freeze_json(destination / 'contract.json', contract)
    global_tables, country_tables, outputs = [], [], []
    for group, physical in physical_partitions(inventory, manifest):
        name = '__'.join(group[field] for field in GROUP_FIELDS)
        details, countries, totals = evaluate_partition(inventory, physical, score_table)
        check_baseline(totals, baseline, group)
        for field, value in group.items():
            countries[field] = value
            totals[field] = value
        path = destination / 'waterbody_partitions' / (name + '.parquet')
        payload = BytesIO()
        details.to_parquet(payload, index=False)
        immutable_bytes(path, payload.getvalue())
        outputs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(details), **group))
        global_tables.append(totals)
        country_tables.append(countries)
        print('evaluated', name, 'scenarios', len(totals), flush=True)
    if len(outputs) != 46:
        raise ValueError('Incomplete current plus future partition set')
    for name, tables in [('global_summary', global_tables), ('country_summary', country_tables)]:
        immutable_bytes(destination / (name + '.csv'), pd.concat(tables, ignore_index=True).to_csv(index=False, lineterminator='\n').encode())
    report = dict(status='conditional_weighted_levels_computed_pending_scientific_review_and_paper',
        all_numerical_checks_passed=True, waterbodies=len(inventory), physical_partitions=46,
        global_scenario_rows=sum(len(t) for t in global_tables),
        checks=['per_waterbody_hierarchy', 'unchanged_physical_L1', 'hard_exclusions', 'cost_gate',
                'zero_strength_physical_identity', 'country_global_sums', 'validated_zero_strength_baseline'],
        contract_sha256=sha(destination / 'contract.json'), waterbody_outputs=outputs,
        summary_hashes={name: sha(destination / name) for name in ['global_summary.csv', 'country_summary.csv']},
        scientific_accuracy_validated=False, paper_updated=False, energy_water_analysis_complete=False,
        note='Conditional scenario outputs, not observed buildable potential. Missing treatments are not confidence intervals.')
    freeze_json(destination / 'validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'waterbody_outputs'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run(args.receipt, args.run_id)
