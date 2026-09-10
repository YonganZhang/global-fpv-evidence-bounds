"""Decompose future changes using NEW fully weighted outputs, never old L3.

Retains the existing ordered decomposition: generation at historical admission,
then admission at future generation. Unknown ice is a separately labelled subset,
not automatically interpreted as thaw. This module makes no new SDK calls.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY
from src.analysis.prepare_global_scores_v124 import GLOBAL, sha, freeze_json, immutable_bytes
from src.analysis.global_level_kernel_v124 import flag
from src.analysis.evaluate_global_levels_v124 import GROUP_FIELDS, ICE_CASES, load_sources
from src.analysis.validate_global_campaign_v124 import validate

KEYS = ['wb_id', 'protection', 'endpoint', 'missing_case']
CONFIG = ['protection', 'endpoint', 'missing_case']
FIXED = ['ecological_area_fraction', 'development_area_fraction', 'engineering_cost_factor',
         'maintenance_cost_factor', 'missing_components']


def admission(frame):
    env, access = flag(frame, 'env_pass'), flag(frame, 'access_pass')
    numbers = frame[['L1_gwh', 'L2_gwh', 'L3_gwh', 'adjusted_cost_usd_mwh', *FIXED]].to_numpy(float)
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError('Invalid weighted numeric output')
    for column in ['ecological_area_fraction', 'development_area_fraction']:
        if frame[column].gt(1).any(): raise ValueError('Invalid area fraction')
    a2 = env * frame.ecological_area_fraction.to_numpy()
    a3 = a2 * access * frame.adjusted_cost_usd_mwh.le(100).to_numpy() * frame.development_area_fraction.to_numpy()
    for level, weight in [('L2', a2), ('L3', a3)]:
        if not np.allclose(frame[level + '_gwh'], frame.L1_gwh * weight, rtol=1e-12, atol=1e-8):
            raise ValueError('Stored levels disagree with admission factors')
    return {'L1': np.ones(len(frame)), 'L2': a2, 'L3': a3}


def ice_known(table, ordered_ids):
    if table.wb_id.isna().any() or not table.wb_id.is_unique or set(table.wb_id) != set(ordered_ids):
        raise ValueError('Ice context does not cover the same waterbodies')
    values = table.set_index('wb_id').loc[ordered_ids, 'ice_duration_days_used'].to_numpy(float)
    if np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 365)).any():
        raise ValueError('Invalid ice duration')
    return np.isfinite(values)


def decompose_pair(historical, future, historical_ice, future_ice):
    """Compute one same-model, same-ice-case pair across the saved configurations."""
    for table in [historical, future]:
        if table.empty or table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError('Empty, duplicate or missing level keys')
        if not table.arm.eq('equal_provider_mean').all() or not table.strength.eq(.5).all():
            raise ValueError('Expected primary equal-provider strength-0.5 outputs')
    h = historical.set_index(KEYS).sort_index()
    f = future.set_index(KEYS).sort_index()
    if not h.index.equals(f.index): raise ValueError('Historical/future population or settings differ')
    h, f = h.reset_index(), f.reset_index()
    if not h.country.equals(f.country) or not np.array_equal(flag(h, 'access_pass'), flag(f, 'access_pass')):
        raise ValueError('Country or infrastructure changed in a fixed-governance experiment')
    if not np.array_equal(h[FIXED].to_numpy(), f[FIXED].to_numpy()):
        raise ValueError('Contemporary LLM area/cost factors changed between periods')
    ah, af = admission(h), admission(f)
    unknown = ~(ice_known(historical_ice, h.wb_id) & ice_known(future_ice, f.wb_id))
    env_changed = flag(h, 'env_pass') != flag(f, 'env_pass')
    cost_changed = h.adjusted_cost_usd_mwh.le(100).to_numpy() != f.adjusted_cost_usd_mwh.le(100).to_numpy()
    out = f[KEYS + ['country']].copy()
    out['ice_unknown_in_either_period'] = unknown
    out['environment_gate_changed'] = env_changed
    out['cost_gate_changed'] = cost_changed
    delta_yield = f.L1_gwh.to_numpy() - h.L1_gwh.to_numpy()
    for level in ['L1', 'L2', 'L3']:
        previous = h[level + '_gwh'].to_numpy()
        following = f[level + '_gwh'].to_numpy()
        generation = delta_yield * ah[level]
        qualification = f.L1_gwh.to_numpy() * (af[level] - ah[level])
        if not np.allclose(following - previous, generation + qualification, rtol=1e-10, atol=1e-7):
            raise ValueError('Change components do not sum to level change')
        out[level + '_historical_gwh'] = previous
        out[level + '_future_gwh'] = following
        out[level + '_change_gwh'] = following - previous
        out[level + '_generation_effect_gwh'] = generation
        out[level + '_admission_effect_gwh'] = qualification
        if level == 'L1': continue
        entry, exit_ = (ah[level] == 0) & (af[level] > 0), (ah[level] > 0) & (af[level] == 0)
        out[level + '_entry'] = entry
        out[level + '_exit'] = exit_
        out[level + '_entry_with_unknown_ice'] = entry & unknown
        out[level + '_admission_effect_unknown_ice_gwh'] = np.where(unknown, qualification, 0.)
        if level == 'L3':
            causes = [env_changed & ~cost_changed, ~env_changed & cost_changed, env_changed & cost_changed]
            for name, cause in zip(['environment_only', 'cost_only', 'both_gates'], causes):
                out['L3_entry_' + name] = entry & cause
            if not np.array_equal(sum((entry & cause).astype(int) for cause in causes), entry.astype(int)):
                raise ValueError('Entry not explained by changing environmental/cost gates')
    return out


def load_weighted_run(run_id):
    if not run_id.isalnum(): raise ValueError('Use an alphanumeric weighted run ID')
    directory = GLOBAL / 'weighted_levels' / run_id
    report_path = directory / 'validation.json'
    report = json.loads(report_path.read_text())
    contract_path = directory / 'contract.json'
    contract = json.loads(contract_path.read_text())
    if (not report['all_numerical_checks_passed'] or report['waterbodies'] != 199976
        or report['physical_partitions'] != 46 or report['contract_sha256'] != sha(contract_path)):
        raise ValueError('Incomplete or changed weighted computation')
    validate(contract['source_receipt'], require_complete=True)
    if sha(ROOT / contract['source_receipt']) != contract['source_receipt_sha256']:
        raise ValueError('Source scoring receipt changed')
    for path, fingerprint in {**contract['source_hashes'], **contract['code_hashes']}.items():
        if sha(ROOT / path) != fingerprint: raise ValueError('Weighted run source/code changed: ' + path)
    manifest, _, _ = load_sources()
    expected = {(r['model'], r['experiment'], '_'.join(map(str, r['period'])), ice) for r in manifest['results'] for ice in ICE_CASES}
    expected.add(('current_reference', 'current', 'POWER_2019_2023', 'current_inventory'))
    lookup = {tuple(ref[k] for k in GROUP_FIELDS): ref for ref in report['waterbody_outputs']}
    if len(lookup) != 46 or len(report['waterbody_outputs']) != 46 or set(lookup) != expected:
        raise ValueError('Weighted partition membership differs from physical design')
    return directory, report, manifest, lookup


def read_weighted(ref, directory, inventory_ids):
    path = (ROOT / ref['path']).resolve()
    if not path.is_relative_to(directory.resolve() / 'waterbody_partitions') or sha(path) != ref['sha256']:
        raise ValueError('Changed or misplaced weighted partition')
    table = pd.read_parquet(path)
    settings = {(p, e, m) for p in ['core', 'conservative'] for e in ['lower', 'upper'] for m in ['unadjusted', 'restricted']}
    if (len(table) != len(inventory_ids) * 8 or ref['rows'] != len(table) or table.duplicated(KEYS).any()
        or set(table.wb_id) != set(inventory_ids) or set(table[CONFIG].itertuples(index=False, name=None)) != settings):
        raise ValueError('Missing, duplicated or misconfigured weighted waterbodies')
    return table


def read_ice(source, ice_case):
    path = ROOT / source['path']
    if sha(path) != source['sha256']: raise ValueError('Future physical source changed')
    data = pd.read_parquet(path, columns=['wb_id', 'ice_duration_days_used'], filters=[('ice_error_case', '=', ice_case)])
    return data


def run(run_id):
    directory, report, manifest, lookup = load_weighted_run(run_id)
    ids = pd.read_parquet(INVENTORY, columns=['wb_id']).wb_id
    sources = {(r['model'], r['experiment'], '_'.join(map(str, r['period']))): r for r in manifest['results']}
    destination = directory / 'future_changes'
    freeze_json(destination / 'contract.json', dict(parent_validation_sha256=sha(directory / 'validation.json'),
        code_sha256=sha(Path(__file__)), comparison='same climate model and ice error case against 1995_2014',
        ordered_decomposition='generation at historical admission, then admission at future generation',
        unknown_ice='reported as subset, not an additional additive component or proof of thaw'))
    globals_, countries_, outputs = [], [], []
    cached_key, cached_history, cached_ice = None, None, None
    for key, ref in lookup.items():
        model, experiment, period, ice_case = key
        if experiment in {'current', 'historical'}: continue
        historical_key = (model, 'historical', '1995_2014', ice_case)
        if cached_key != historical_key:
            cached_history = read_weighted(lookup[historical_key], directory, ids)
            cached_ice = read_ice(sources[historical_key[:3]], ice_case)
            cached_key = historical_key
        future = read_weighted(ref, directory, ids)
        change = decompose_pair(cached_history, future, cached_ice, read_ice(sources[key[:3]], ice_case))
        group = dict(zip(GROUP_FIELDS, key))
        measures = [c for c in change if c not in KEYS + ['country']]
        countries = change.groupby(CONFIG + ['country'], sort=True)[measures].sum().reset_index()
        totals = change.groupby(CONFIG, sort=True)[measures].sum().reset_index()
        regrouped = countries.groupby(CONFIG)[measures].sum().sort_index()
        if not np.allclose(regrouped.to_numpy(float), totals.set_index(CONFIG).sort_index()[measures].to_numpy(float), rtol=1e-10, atol=1e-6):
            raise ValueError('Country/global future changes do not reconcile')
        for column, value in group.items():
            countries[column], totals[column] = value, value
        path = destination / 'waterbodies' / ('__'.join(key) + '.parquet')
        payload = BytesIO()
        change.to_parquet(payload, index=False)
        immutable_bytes(path, payload.getvalue())
        outputs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(change), **group))
        countries_.append(countries)
        globals_.append(totals)
        print('decomposed', '__'.join(key), flush=True)
    if len(outputs) != 36: raise ValueError('Incomplete future comparison set')
    for name, tables in [('country_changes', countries_), ('global_changes', globals_)]:
        immutable_bytes(destination / (name + '.csv'), pd.concat(tables, ignore_index=True).to_csv(index=False, lineterminator='\n').encode())
    result = dict(status='conditional_future_changes_not_future_governance', all_checks_passed=True,
        waterbodies=199976, paired_partitions=36, global_comparison_rows=sum(len(t) for t in globals_),
        outputs=outputs, contract_sha256=sha(destination / 'contract.json'),
        summary_hashes={name: sha(destination / name) for name in ['country_changes.csv', 'global_changes.csv']},
        unknown_ice_separated=True, fixed_factors_checked=True, scientific_accuracy_validated=False,
        note='Entry/exit are scenario eligibility changes, not observed projects. Unknown-ice effects are subsets, not extra additive causes.')
    freeze_json(destination / 'validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'outputs'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args().run_id)
