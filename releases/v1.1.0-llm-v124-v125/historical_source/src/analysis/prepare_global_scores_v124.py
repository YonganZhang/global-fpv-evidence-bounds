"""Freeze score contexts for ALL waterbodies, without selecting on present yield.

Missing context stays missing. This is an input map, not a potential estimate.
Existing pilot IDs/prompts are preserved wherever context matches exactly.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, OUT, INVENTORY, TAGS, build_entity, load_inventory

GLOBAL = OUT / 'global'
ASSESSMENT_DATE = '2026-09-08'
CONTEXT_FIELDS = {
    'ecology': {'waterbody_type':'wb_type', 'terrestrial_ecoregion':'ecoregion', 'biome':'biome'},
    'engineering': {'waterbody_type':'wb_type', 'koppen_zone':'koppen_zone', 'country':'country_key'},
    'country': {'country':'country_key'},
    'province': {'country':'country_key', 'province':'province'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_bytes(path, payload):
    """Allow identical replay; refuse to overwrite different frozen input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Frozen artifact changed: ' + str(path))
        return
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('xb') as handle:
        handle.write(payload)
    os.link(temporary, path)
    temporary.unlink()


def freeze_json(path, value):
    immutable_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def present(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {'nan','none','null','unknown','n/a'}


def make_global_map(frame):
    if frame.empty or not frame.wb_id.is_unique or frame.wb_id.isna().any():
        raise ValueError('Empty inventory or duplicate/missing waterbody ID')
    mapping = frame[['wb_id']].copy()
    entities = {}
    for dimension, fields in CONTEXT_FIELDS.items():
        columns = list(fields.values())
        valid = frame[columns].map(present).all(axis=1)
        if 'wb_type' in columns:
            valid &= frame.wb_type.isin(['lake','reservoir'])
        # Historical regional tags require explicit current-country agreement.
        if dimension != 'country':
            valid &= frame.tags_country_match.eq(True)
        assigned = pd.Series(None, index=frame.index, dtype=object)
        unique = frame.loc[valid, columns].drop_duplicates()
        keys = {}
        for values in unique.itertuples(index=False, name=None):
            context = dict(zip(fields, values))
            if dimension in {'country','province'}:
                context['assessment_date'] = ASSESSMENT_DATE
            entity = build_entity(dimension, context)
            prior = entities.setdefault(entity['entity_id'], entity)
            if prior != entity:
                raise ValueError('Entity hash collision')
            keys[values] = entity['entity_id']
        assigned.loc[valid] = [keys[v] for v in frame.loc[valid,columns].itertuples(index=False,name=None)]
        mapping[dimension+'_entity_id'] = assigned
    # Sorting is independent of energy, cost, suitability or model scores.
    entities = sorted(entities.values(), key=lambda x:(list(CONTEXT_FIELDS).index(x['dimension']),x['entity_id']))
    return mapping.sort_values('wb_id').reset_index(drop=True), entities


def prepare():
    frame = load_inventory()
    mapping, entities = make_global_map(frame)
    checks = {
        'full_inventory_membership': len(mapping)==len(frame) and set(mapping.wb_id)==set(frame.wb_id),
        'unique_entities': len({e['entity_id'] for e in entities})==len(entities),
        'identities_reconstruct': all(build_entity(e['dimension'],e['context'])==e for e in entities),
        'no_incomplete_context_in_prompts': all(all(present(v) for v in e['context'].values()) for e in entities),
    }
    lookup = {e['entity_id']:e for e in entities}
    pilot = json.loads((OUT/'data/pilot_entities.json').read_text())
    checks['all_pilot_entities_exactly_reusable'] = all(lookup.get(e['entity_id'])==e for e in pilot)
    if not all(checks.values()):
        raise ValueError(checks)
    # CSV bytes are stable, including blank (not fabricated) missing IDs.
    immutable_bytes(GLOBAL/'waterbody_entity_map.csv', mapping.to_csv(index=False,lineterminator='\n').encode())
    freeze_json(GLOBAL/'entities.json', entities)
    summary = dict(status='global_contexts_frozen_not_scored', waterbodies=len(mapping),
        entities=len(entities), required_predictions=len(entities)*4,
        dimension_entities={d:sum(e['dimension']==d for e in entities) for d in CONTEXT_FIELDS},
        missing_context_waterbodies={d:int(mapping[d+'_entity_id'].isna().sum()) for d in CONTEXT_FIELDS},
        checks=checks, all_checks_passed=all(checks.values()), assessment_date=ASSESSMENT_DATE,
        selection='All inventory IDs, including currently excluded sites; no current L2/L3 gate used.',
        future_rule='Contemporary institutional snapshot held fixed; future physical gates and costs must be rebuilt, not multiplied onto v123.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [INVENTORY,TAGS,Path(__file__),ROOT/'src/analysis/fpv_llm_v124.py']},
        artifact_hashes={name:sha(GLOBAL/name) for name in ['entities.json','waterbody_entity_map.csv']},
        global_result=False, scientific_accuracy_validated=False)
    freeze_json(GLOBAL/'preparation.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    prepare()
