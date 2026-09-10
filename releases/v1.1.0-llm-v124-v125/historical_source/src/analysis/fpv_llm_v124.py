"""Separated, direct LM judgments: preparation and conditional L2/L3 consumer.

Pilot only until coverage and scientific validation are approved. Scores are not
observed data, permission decisions, measured costs or calibrated probabilities.
Run with the future virtualenv; no import of the historical API evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.future.align_llm_scores import country_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '_outputs/v124'
INVENTORY = ROOT / '_outputs/v117/data/fpv_reference_inventory_v117.parquet'
TAGS = ROOT / 'Dataset/data_processed/merged/waterbody_tags_enriched.csv'
RAW = ROOT / 'Dataset/data_processed/llm_scores'
DIMENSIONS = {
    'country': ('dim_1a_country_policy.csv', 's_policy_overall', 0., 1.),
    'province': ('dim_1b_province_modifier.csv', 's_province_modifier', .5, 1.5),
    'climate': ('dim_2b_climate_modifier.csv', 's_climate_llm_modifier', 0., 1.),
    'ecology': ('dim_3_ecology_score.csv', 's_ecology_score', 0., 1.),
    'named': ('dim_6_waterbody_specific.csv', 's_specific', 0., 1.),
}
FIELDS = {
    'ecology': {'ecological_suitability': (0., 1.)},
    'engineering': {'maintenance_ease': (0., 1.)},
    'country': {'renewable_support': (0., 1.), 'finance_delivery': (0., 1.)},
    'province': {'implementation_offset': (-.15, .15)},
}


def validate_number(value, lo, hi):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and lo <= value <= hi


def load_model_values(row, field, lo, hi):
    try:
        models = json.loads(row.get('_per_model', '{}'))
    except (TypeError, ValueError):
        models = {}
    if not isinstance(models, dict):
        models = {}
    values = []
    for item in models.values():
        if not isinstance(item, dict) or item.get('status') != 'ok':
            continue
        scores = item.get('scores')
        value = scores.get(field) if isinstance(scores, dict) else None
        if validate_number(value, lo, hi):
            values.append(float(value))
    return values


def load_raw_shape(row):
    try:
        value = json.loads(row.get('_per_model', '{}'))
    except (TypeError, ValueError):
        return 'invalid_json'
    return 'model_dictionary' if isinstance(value, dict) else 'wrong_payload_type'


def compute_trimmed_mean(values):
    values = sorted(values)
    if len(values) >= 4:
        values = values[1:-1]
    return round(sum(values) / len(values), 3) if values else None


def build_raw_audit():
    """Retain failed, zero and missing states; never impute an aggregate."""
    records, source_hashes = [], {}
    for dimension, (filename, field, lo, hi) in DIMENSIONS.items():
        path = RAW / filename
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = pd.read_csv(path)
        for row_id, row in enumerate(data.to_dict('records')):
            values = load_model_values(row, field, lo, hi)
            stored = row.get(field)
            rebuilt = compute_trimmed_mean(values)
            valid = validate_number(stored, lo, hi)
            matches = rebuilt is not None and valid and abs(stored - rebuilt) < 1e-9
            shape = load_raw_shape(row)
            state = ('unrecoverable_model_record' if shape != 'model_dictionary'
                     else 'no_valid_model_output' if not values else 'invalid_stored_score' if not valid
                     else 'aggregation_mismatch' if not matches else 'available_unvalidated')
            records.append(dict(dimension=dimension, row_id=row_id, entity_key=row['entity_key'],
                field=field, stored_score=stored, reconstructed_score=rebuilt,
                valid_models=len(values), state=state, per_model_shape=shape,
                model_min=min(values) if values else None, model_max=max(values) if values else None,
                zero_models=values.count(0.), declared_models=row.get('_ensemble_n')))
    return pd.DataFrame(records), source_hashes


def load_inventory():
    frame, tags = pd.read_parquet(INVENTORY), pd.read_csv(TAGS)
    if not frame.wb_id.is_unique or not tags.wb_id.is_unique:
        raise ValueError('Non-unique waterbody IDs')
    tags = tags.rename(columns={'country': 'historical_country'})
    frame = frame.merge(tags, on='wb_id', how='left', validate='one_to_one')
    frame['country_key'] = frame.country_v117.fillna('').map(country_key)
    frame['historical_country_key'] = frame.historical_country.fillna('').map(country_key)
    frame['tags_country_match'] = frame.country_key.eq(frame.historical_country_key) & frame.country_key.ne('')
    # All historical categorical assignments require matched country. This does
    # not establish current province/ecoregion accuracy; it only rejects stale joins.
    for field in ['province', 'koppen_zone', 'ecoregion', 'biome']:
        frame.loc[~frame.tags_country_match, field] = None
    return frame


def build_masks(frame):
    def flag(field):
        value = frame[field]
        if value.isna().any() or not value.isin([True, False]).all():
            raise ValueError('Missing/nonboolean gate: ' + field)
        return value.astype(bool)
    env = flag('l1_area_pass_v112') & flag('ice_pass_v112') & ~flag('in_wdpa_polygon_strict_i_iv')
    # Inventory encodes unavailable drying history as NA, not known=False.
    # Absence of knowledge belongs only to the upper conditional endpoint.
    known = frame.historical_dryup_evidence_known_v113.fillna(False).astype(bool)
    no_dry = frame.historical_no_dryup_observed_1991_2018_v113.fillna(False).astype(bool)
    if frame.loc[known, 'historical_no_dryup_observed_1991_2018_v113'].isna().any():
        raise ValueError('Known drying history lacks its outcome')
    result = {}
    for end, dry in [('lower', known & no_dry), ('upper', ~known | no_dry)]:
        result['L2_' + end] = env & dry
        road = 'road_within_10km_' + ('guaranteed' if end == 'lower' else 'possible') + '_v115'
        grid = 'grid_within_25km_' + ('observed' if end == 'lower' else 'possible') + '_or_hydropower_v115'
        # Population is a retained development proxy, no longer an environmental gate.
        result['access_' + end] = flag(road) & flag(grid) & flag('within_10km_population_center')
    return result


def build_entity(dimension, context):
    payload = dict(dimension=dimension, context=context)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return dict(entity_id=dimension + '_' + key, **payload)


def build_pilot(frame):
    masks = build_masks(frame)
    selected = []
    # Deterministic eligible cases for a wiring test, NOT a representative sample.
    for country, kind in [('China', 'reservoir'), ('China', 'lake'), ('Canada', 'lake'), ('Brazil', 'reservoir')]:
        eligible = (frame.country_key.eq(country) & frame.wb_type.eq(kind) & frame.tags_country_match
                    & frame[['province', 'koppen_zone', 'ecoregion', 'biome']].notna().all(axis=1)
                    & masks['L2_upper'] & masks['access_upper']
                    & frame.lcoe_fpv_reference_usd_mwh_v117.le(100))
        sample = frame.loc[eligible].sort_values(['area_km2', 'wb_id'], ascending=[False, True]).head(1)
        if len(sample) != 1:
            raise ValueError('No eligible pilot case: ' + country + ' ' + kind)
        selected.append(sample)
    cases = pd.concat(selected).copy()
    entities = {}
    for index, row in cases.iterrows():
        contexts = {
            'ecology': dict(waterbody_type=row.wb_type, terrestrial_ecoregion=row.ecoregion, biome=row.biome),
            'engineering': dict(waterbody_type=row.wb_type, koppen_zone=row.koppen_zone, country=row.country_key),
            'country': dict(country=row.country_key, assessment_date='2026-09-08'),
            'province': dict(country=row.country_key, province=row.province, assessment_date='2026-09-08'),
        }
        for dim, context in contexts.items():
            entity = build_entity(dim, context)
            entities[entity['entity_id']] = entity
            cases.loc[index, dim + '_entity_id'] = entity['entity_id']
    return cases, list(entities.values())


def validate_response(dimension, value):
    if not isinstance(value, dict) or set(value) != set(FIELDS[dimension]):
        raise ValueError('Wrong score fields')
    for field, (lo, hi) in FIELDS[dimension].items():
        item = value[field]
        if not isinstance(item, dict) or set(item) != {'score', 'confidence', 'reason'}:
            raise ValueError('Wrong response structure')
        if item['score'] is not None and not validate_number(item['score'], lo, hi):
            raise ValueError('Out-of-range/non-numeric score')
        if item['confidence'] not in {'low', 'medium', 'high', 'unknown'}:
            raise ValueError('Invalid confidence')
        if item['confidence'] == 'unknown' and item['score'] is not None:
            raise ValueError('Unknown confidence requires null score')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Missing rationale')
    return value


def compute_levels(l1, env_pass, access_pass, cost, ecology, maintenance, support,
                   finance, province_offset, strength=.5, missing_case='unadjusted'):
    """Scalar pilot consumer. Prespecified strength is a scenario, not fitted.

    Ecology and institutional weights mean hypothetical usable-area fractions.
    Engineering changes the reference cost index, never PV conversion efficiency.
    Unknown treatments are neutral/restricted scenarios, NOT exhaustive bounds
    or a confidence interval (a missing province can also be favorable).
    """
    if not validate_number(l1, 0, 1e12) or not validate_number(cost, 0, 1e12):
        raise ValueError('Invalid base energy/cost')
    if not validate_number(strength, 0, 1) or missing_case not in {'unadjusted', 'restricted'}:
        raise ValueError('Invalid scenario')
    for value in [env_pass, access_pass]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('Gates must be boolean')
    scores = [ecology, maintenance, support, finance]
    if any(v is not None and not validate_number(v, 0, 1) for v in scores):
        raise ValueError('Invalid LM score')
    if province_offset is not None and not validate_number(province_offset, -.15, .15):
        raise ValueError('Invalid province offset')
    missing = sum(v is None for v in scores) + (province_offset is None)
    e, m, s, f = [v if v is not None else (1. if missing_case == 'unadjusted' else 0.) for v in scores]
    # Missing provincial context is not an inferred zero-valued judgment.
    p = province_offset if province_offset is not None else (0. if missing_case == 'unadjusted' else -.15)
    dev = min(1., max(0., (s + f) / 2 + p))
    eco_fraction = 1 - strength * (1 - e)
    dev_fraction = 1 - strength * (1 - dev)
    # Only maintenance/cleaning difficulty was elicited: modify discounted O&M,
    # never the capital cost. Match v117 reference: O&M=2% CAPEX, WACC=8%, 25 y.
    om_pv_per_capex = .02 * sum(1 / 1.08 ** year for year in range(1, 26))
    om_share = om_pv_per_capex / (1 + om_pv_per_capex)
    maintenance_factor = 1 + strength * (1 - m)
    cost_factor = 1 + om_share * (maintenance_factor - 1)
    adjusted_cost = cost * cost_factor
    l2 = l1 * bool(env_pass) * eco_fraction
    l3 = l2 * bool(access_pass) * (adjusted_cost <= 100.) * dev_fraction
    return dict(L1_gwh=l1, L2_gwh=l2, L3_gwh=l3, adjusted_cost_usd_mwh=adjusted_cost,
                ecological_area_fraction=eco_fraction, development_area_fraction=dev_fraction,
                engineering_cost_factor=cost_factor, maintenance_cost_factor=maintenance_factor,
                discounted_om_share=om_share, missing_components=missing)


def run_prepare():
    for directory in [OUT / 'data', OUT / 'reports']:
        directory.mkdir(parents=True, exist_ok=True)
    audit, hashes = build_raw_audit()
    frame = load_inventory()
    masks = build_masks(frame)
    cases, entities = build_pilot(frame)
    audit.to_csv(OUT / 'reports/raw_score_audit.csv', index=False)
    cases.to_parquet(OUT / 'data/pilot_cases.parquet', index=False)
    (OUT / 'data/pilot_entities.json').write_text(json.dumps(entities, ensure_ascii=False, indent=2) + '\n')
    # Reclassification alone must preserve L3 before any additional LM factors.
    changed = {}
    for end in ['lower', 'upper']:
        old = frame.l2_core_no_dry_pass_v112.astype(bool)
        known = frame.historical_dryup_evidence_known_v113.fillna(False).astype(bool)
        no_dry = frame.historical_no_dryup_observed_1991_2018_v113.fillna(False).astype(bool)
        old = old & ((known & no_dry) if end == 'lower' else (~known | no_dry))
        if not (old & masks['access_' + end]).equals(masks['L2_' + end] & masks['access_' + end]):
            raise ValueError('Population reclassification changed L3 baseline')
        changed[end] = int((masks['L2_' + end] & ~old).sum())
    summary = dict(status='inputs_prepared_not_global_results', waterbodies=len(frame),
        rejected_historical_tag_rows=int((~frame.tags_country_match).sum()),
        historical_score_states=audit.groupby(['dimension', 'state']).size().rename('rows').reset_index().to_dict('records'),
        pure_environment_additional_candidates=changed, population_moved_to_L3=True,
        pilot_waterbodies=len(cases), pilot_entities=len(entities),
        sample_selection='Largest eligible case per fixed country/type stratum; wiring test, not representative',
        source_hashes={**hashes, **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [INVENTORY, TAGS, Path(__file__)]}},
        limitations=['Old mixed climate/ecology/named scores are not reused.',
            'Historical territorial tags remain proxies, not site observations.',
            'No global weighted result or scientific validity is asserted.'])
    (OUT / 'reports/preparation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_prepare()
