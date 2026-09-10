"""Freeze the approved Codex-primary/Claude-audit design without scoring calls.

@role: entry
@produces: _outputs/v124/global/lean/plan.json
The sampling rule uses the full inventory's geography, never scores or eligibility.
This is a retrospective protocol revision: prior responses already existed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY, FIELDS, build_entity
from src.analysis.prepare_global_scores_v124 import GLOBAL, sha, freeze_json
from src.analysis.run_global_subscription_v124 import prompt_ordered_entity

LEAN = GLOBAL / 'lean'
SOURCE = GLOBAL / 'runs/global05/receipts/20260908T170844090574.json'
SEED = 'fpv-primary-audit-20260909-v1'


def allocate(sizes, target, minimum_one=False):
    """Integer largest remainders, optionally after reserving one per stratum."""
    if (type(target) is not int or any(type(v) is not int or v < 0 for v in sizes.values())
        or not 0 <= target <= sum(sizes.values())):
        raise ValueError('Invalid sampling capacity or target')
    counts = {k: int(minimum_one and v > 0) for k, v in sizes.items()}
    remaining = target - sum(counts.values())
    if remaining < 0: raise ValueError('Too many strata for requested sample')
    capacities = {k: sizes[k] - counts[k] for k in sizes}
    total = sum(capacities.values())
    if not total: return counts
    extra = {k: capacities[k] * remaining // total for k in sizes}
    residual = remaining - sum(extra.values())
    order = sorted(sizes, key=lambda k: (-(capacities[k] * remaining % total), k))
    for k in order[:residual]: extra[k] += 1
    return {k: counts[k] + extra[k] for k in sizes}


def sample_entities(entities, mapping, inventory, sample_size=471):
    lookup = {e['entity_id']: e for e in entities}
    if len(lookup) != len(entities) or not entities:
        raise ValueError('Invalid entity catalogue')
    for frame in [mapping, inventory]:
        if frame.wb_id.isna().any() or not frame.wb_id.is_unique:
            raise ValueError('Invalid waterbody keys')
    if set(mapping.wb_id) != set(inventory.wb_id):
        raise ValueError('Geography does not cover the complete mapping')
    geo = inventory.set_index('wb_id').loc[mapping.wb_id, 'continent_v117'].reset_index(drop=True)
    geo = geo.astype('string').str.strip().fillna('Unknown').replace('', 'Unknown')
    region_by_entity = {}
    for dimension in FIELDS:
        column = dimension + '_entity_id'
        part = pd.DataFrame({'entity_id': mapping[column].to_numpy(), 'region': geo.to_numpy()}).dropna(subset=['entity_id'])
        expected = {e['entity_id'] for e in entities if e['dimension'] == dimension}
        if set(part.entity_id) != expected: raise ValueError('Unused or cross-dimension mapped entity')
        for entity_id, group in part.groupby('entity_id', sort=True):
            # A cross-continent region remains a labelled set; no majority-country guessing.
            region_by_entity[entity_id] = '+'.join(sorted(set(group.region)))
    groups = defaultdict(list)
    for e in entities:
        kind = e['context'].get('waterbody_type', 'all')
        groups[(e['dimension'], region_by_entity[e['entity_id']], kind)].append(e['entity_id'])
    dimensions = dict(Counter(e['dimension'] for e in entities))
    quotas = allocate(dimensions, sample_size)
    strata, selected = [], []
    for dimension in sorted(dimensions):
        sizes = {k: len(v) for k, v in groups.items() if k[0] == dimension}
        assigned = allocate(sizes, quotas[dimension], minimum_one=True)
        for key in sorted(sizes):
            ids = sorted(groups[key])
            ordered = sorted(ids, key=lambda eid: (hashlib.sha256((SEED + '|' + eid).encode()).hexdigest(), eid))
            chosen = ordered[:assigned[key]]
            selected.extend(chosen)
            strata.append(dict(dimension=key[0], region=key[1], waterbody_type=key[2], entity_ids=ids,
                selected_ids=chosen, population=len(ids), sample_size=len(chosen),
                inclusion_probability=len(chosen) / len(ids)))
    if len(selected) != sample_size or len(set(selected)) != sample_size:
        raise ValueError('Audit sample size mismatch')
    return sorted(selected), quotas, strata


def expected_slots(entities, audit_ids):
    audit = set(audit_ids)
    if not audit <= {e['entity_id'] for e in entities}: raise ValueError('Unknown audit entity')
    slots = []
    for e in entities:
        slots.append(dict(entity_id=e['entity_id'], provider='codex', repeat=0, role='primary'))
        if e['entity_id'] in audit:
            slots.append(dict(entity_id=e['entity_id'], provider='claude', repeat=0, role='audit'))
    return slots


def prepare_inputs():
    preparation = json.loads((GLOBAL / 'preparation.json').read_text())
    if not preparation['all_checks_passed'] or preparation['waterbodies'] != 199976 or preparation['entities'] != 4701:
        raise ValueError('Wrong frozen population')
    for name, fingerprint in preparation['artifact_hashes'].items():
        if sha(GLOBAL / name) != fingerprint: raise ValueError('Frozen preparation artifact changed')
    if sha(INVENTORY) != preparation['source_hashes'][str(INVENTORY.relative_to(ROOT))]:
        raise ValueError('Physical inventory changed')
    entities = [prompt_ordered_entity(e) for e in json.loads((GLOBAL / 'entities.json').read_text())]
    if any(build_entity(e['dimension'], e['context']) != e for e in entities):
        raise ValueError('Entity identity changed')
    mapping = pd.read_csv(GLOBAL / 'waterbody_entity_map.csv')
    inventory = pd.read_parquet(INVENTORY, columns=['wb_id', 'continent_v117'])
    return entities, mapping, inventory


def build_plan():
    entities, mapping, inventory = prepare_inputs()
    audit, quotas, strata = sample_entities(entities, mapping, inventory)
    source = json.loads(SOURCE.read_text())
    contract = SOURCE.parent.parent / 'contract.json'
    if source['contract_sha256'] != sha(contract): raise ValueError('Legacy source contract changed')
    sources = [SOURCE, contract, GLOBAL / 'preparation.json', GLOBAL / 'entities.json',
               GLOBAL / 'waterbody_entity_map.csv', INVENTORY, Path(__file__)]
    plan = dict(schema_version='fpv-primary-audit-v1', source_receipt=str(SOURCE.relative_to(ROOT)),
        source_receipt_sha256=sha(SOURCE), source_contract_sha256=sha(contract),
        preparation_sha256=sha(GLOBAL / 'preparation.json'), entities_sha256=sha(GLOBAL / 'entities.json'),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
        seed=SEED, primary_provider='codex', audit_provider='claude', canonical_repeat=0,
        primary_count=4701, audit_count=471, expected_predictions=5172, waterbodies=199976,
        dimension_quotas=quotas, audit_entity_ids=audit, strata=strata,
        required_slots=expected_slots(entities, audit),
        primary_aggregation='Codex canonical repeat 0 only for every entity; audit never fills or averages primary',
        selection='Dimension quotas then full-inventory continent-set/type strata; minimum one per nonempty stratum, largest remainders, fixed hash order',
        retrospective=True, prior_scores_used_for_sample_selection=False,
        auxiliary_policy='Legacy repeat1 and unselected Claude remain archived and are not required primary/audit members',
        retry_policy='Reuse accepted canonical0 including null/low; max one combined runtime/format retry per original slot, no third attempt',
        interpretation='Model agreement is not scientific accuracy; oversized stratum sampling needs design weights for pooled estimates')
    return plan


def load_plan():
    plan = json.loads((LEAN / 'plan.json').read_text())
    for path, fingerprint in plan['source_hashes'].items():
        if sha(ROOT / path) != fingerprint: raise ValueError('Reduced plan source changed: ' + path)
    if plan != build_plan(): raise ValueError('Reduced sampling/membership contract changed')
    return plan


def main():
    plan = build_plan()
    freeze_json(LEAN / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in {'strata', 'required_slots', 'source_hashes', 'audit_entity_ids'}}, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
