"""Complete-score-gated present benefits; never a future water/demand forecast."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY
from src.analysis.prepare_global_scores_v124 import sha, freeze_json, immutable_bytes
from src.analysis.analyze_global_future_v124 import load_weighted_run, read_weighted, CONFIG
from src.analysis.global_benefit_metrics_v124 import demand_by_iso, present_benefits

EVAPORATION = ROOT / 'Dataset/data_processed/evaporation/evaporation_savings.csv'
DEMAND = ROOT / 'Dataset/data_processed/IEA electricity demand data/final_consumption_total_2023.csv'
ISO = ROOT / 'Dataset/data_processed/IEA electricity demand data/Country ISO.csv'


def run(run_id):
    directory, _, _, lookup = load_weighted_run(run_id)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [INVENTORY, EVAPORATION, DEMAND, ISO]}
    inventory = pd.read_parquet(INVENTORY)
    weighted = read_weighted(lookup[('current_reference', 'current', 'POWER_2019_2023', 'current_inventory')],
                             directory, inventory.wb_id)
    evaporation = pd.read_csv(EVAPORATION, usecols=['wb_id', 'annual_evap_mm'])
    demand = demand_by_iso(pd.read_csv(DEMAND), pd.read_csv(ISO))
    water, country, total = present_benefits(weighted, inventory, evaporation, demand)
    checks = dict(waterbody_rows=len(water) == 199976 * 8, scenario_totals=len(total) == 8)
    for level in ['L1', 'L2', 'L3']:
        original = weighted.groupby(CONFIG)[level + '_gwh'].sum().sort_index() / 1000.
        result = total.set_index(CONFIG)[level + '_generation_twh'].sort_index()
        checks[level + '_global_energy_reconciles'] = bool(np.allclose(original, result, rtol=1e-10, atol=1e-8))
    if not all(checks.values()): raise ValueError(checks)
    for path, digest in source_hashes.items():
        if sha(ROOT / path) != digest: raise ValueError('Benefit source changed during read')
    destination = directory / 'present_benefits'
    freeze_json(destination / 'contract.json', dict(parent_validation_sha256=sha(directory / 'validation.json'),
        source_hashes=source_hashes,
        code_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / 'src/analysis/global_benefit_metrics_v124.py', ROOT / 'src/analysis/analyze_global_future_v124.py']},
        demand_year=2023, comparison='annual generation versus matched national final electricity consumption',
        evaporation='legacy historical Penman depth, coverage-limited, not future hydrology',
        reduction_on_covered_area=.46, same_level_area=True,
        missing_evaporation='NaN, including wholly unknown country; global savings restricted to known subset'))
    payload = BytesIO()
    water.to_parquet(payload, index=False)
    immutable_bytes(destination / 'waterbody_benefits.parquet', payload.getvalue())
    for name, table in [('country_benefits', country), ('global_benefits', total)]:
        immutable_bytes(destination / (name + '.csv'), table.to_csv(index=False, lineterminator='\n').encode())
    result = dict(status='conditional_present_benefits_not_realized_supply_or_future_hydrology',
        all_checks_passed=True, checks=checks, waterbodies=199976, scenario_totals=8,
        evaporation_known_waterbodies=int(water.loc[water.evaporation_known, 'wb_id'].nunique()),
        demand_source_countries=len(demand), contract_sha256=sha(destination / 'contract.json'),
        output_hashes={name: sha(destination / name) for name in [
            'waterbody_benefits.parquet', 'country_benefits.csv', 'global_benefits.csv']},
        scientific_accuracy_validated=False,
        note='46% is an evaporation-reduction assumption, not an observed global water saving. Annual demand ratios do not imply reliable self-sufficiency.')
    freeze_json(destination / 'validation.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args().run_id)
