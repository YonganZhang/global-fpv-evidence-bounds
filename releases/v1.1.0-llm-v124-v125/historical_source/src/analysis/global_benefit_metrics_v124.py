"""Present-day benefits on exactly the same L1/L2/L3 scenario areas.

Pure numerical consumers: callers must supply verified current weighted tables.
No SDK calls; no future demand or evaporation prediction; no city autonomy claim.
"""
import numpy as np
import pandas as pd

from src.analysis.analyze_global_future_v124 import admission, CONFIG, KEYS


def unique_ids(frame, key):
    if frame[key].isna().any() or not frame[key].is_unique:
        raise ValueError('Missing or duplicate key: ' + key)


def demand_by_iso(demand, country_iso):
    """Map 2023 national final consumption from TJ to TWh, without fuzzy joins."""
    demand, country_iso = demand.copy(), country_iso.copy()
    demand['Country'] = demand.Country.astype('string').str.strip()
    country_iso['Country name'] = country_iso['Country name'].astype('string').str.strip()
    unique_ids(demand, 'Country')
    unique_ids(country_iso, 'Country name')
    joined = demand.merge(country_iso[['Country name', 'ISO']], left_on='Country',
                          right_on='Country name', how='left', validate='one_to_one')
    unique_ids(joined, 'ISO')
    values = pd.to_numeric(joined['TotalConsumption(TJ)'], errors='raise') / 3600.
    if not np.isfinite(values).all() or values.le(0).any():
        raise ValueError('National consumption must be positive and finite')
    return joined[['ISO', 'Country']].assign(electricity_final_consumption_twh=values)


def present_benefits(weighted, inventory, evaporation, demand):
    """Return per-waterbody, per-ISO, and global tables; preserve absent evaporation."""
    unique_ids(inventory, 'wb_id')
    unique_ids(evaporation, 'wb_id')
    unique_ids(demand, 'ISO')
    if weighted.empty or weighted[KEYS].isna().any().any() or weighted.duplicated(KEYS).any():
        raise ValueError('Invalid weighted scenario keys')
    if set(weighted.wb_id) != set(inventory.wb_id):
        raise ValueError('Weighted and physical populations differ')
    for _, group in weighted.groupby(CONFIG):
        if len(group) != len(inventory): raise ValueError('Incomplete scenario population')
    if not weighted.arm.eq('equal_provider_mean').all() or not weighted.strength.eq(.5).all():
        raise ValueError('Expected primary weighted scenarios')
    factors = admission(weighted)
    physical = inventory[['wb_id', 'country_iso3_v117', 'l0_type_specific_generation_gwh',
                          'annual_footprint_energy_kwh_m2']].merge(
        evaporation[['wb_id', 'annual_evap_mm']], on='wb_id', how='left', validate='one_to_one')
    physical = physical.set_index('wb_id').loc[weighted.wb_id].reset_index()
    if not np.allclose(weighted.L1_gwh, physical.l0_type_specific_generation_gwh, rtol=1e-12, atol=1e-8):
        raise ValueError('Expected current generation, not future generation')
    yield_ = physical.annual_footprint_energy_kwh_m2.to_numpy(float)
    if not np.isfinite(yield_).all() or (yield_ <= 0).any():
        raise ValueError('Invalid footprint generation for area reconstruction')
    depth = physical.annual_evap_mm.to_numpy(float)
    if np.isinf(depth).any() or (depth[np.isfinite(depth)] < 0).any():
        raise ValueError('Invalid evaporation depth')
    known = np.isfinite(depth)
    area = physical.l0_type_specific_generation_gwh.to_numpy() / yield_
    output = weighted[KEYS].reset_index(drop=True).copy()
    output['ISO'] = physical.country_iso3_v117
    output['evaporation_known'] = known
    for level in ['L1', 'L2', 'L3']:
        output[level + '_generation_twh'] = weighted[level + '_gwh'].to_numpy() / 1000.
        output[level + '_covered_area_km2'] = area * factors[level]
        output[level + '_water_saving_km3'] = depth * area * factors[level] * .46 / 1e6
        output[level + '_evap_known_generation_twh'] = np.where(known, weighted[level + '_gwh'].to_numpy() / 1000., 0.)
    measures = [c for c in output if c.endswith(('_twh', '_km2', '_km3'))]
    # min_count=1 keeps an entirely unknown jurisdiction's savings unknown.
    countries = output.groupby(CONFIG + ['ISO'], dropna=False)[measures].sum(min_count=1).reset_index()
    countries = countries.merge(demand, on='ISO', how='left', validate='many_to_one')
    observed = countries.electricity_final_consumption_twh
    if ((observed.dropna() <= 0).any() or np.isinf(observed).any()):
        raise ValueError('Invalid demand denominator')
    for level in ['L1', 'L2', 'L3']:
        countries[level + '_demand_ratio_pct'] = countries[level + '_generation_twh'] / observed * 100.
    totals = []
    for config, table in countries.groupby(CONFIG):
        row = dict(zip(CONFIG, config))
        matched = table.electricity_final_consumption_twh.notna()
        row['countries_with_demand'] = int(matched.sum())
        denominator = table.loc[matched, 'electricity_final_consumption_twh'].sum(min_count=1)
        row['matched_final_consumption_twh'] = denominator
        for level in ['L1', 'L2', 'L3']:
            energy = level + '_generation_twh'
            row[energy] = table[energy].sum(min_count=1)
            row[level + '_matched_generation_twh'] = table.loc[matched, energy].sum(min_count=1)
            row[level + '_matched_demand_ratio_pct'] = row[level + '_matched_generation_twh'] / denominator * 100.
            for threshold in [10, 50, 100]:
                row[level + '_countries_ge_' + str(threshold) + 'pct'] = int((matched & table[level + '_demand_ratio_pct'].ge(threshold)).sum())
            row[level + '_water_saving_known_subset_km3'] = table[level + '_water_saving_km3'].sum(min_count=1)
            row[level + '_evap_generation_coverage_pct'] = (
                table[level + '_evap_known_generation_twh'].sum() / row[energy] * 100. if row[energy] else np.nan)
        totals.append(row)
    return output, countries, pd.DataFrame(totals)
