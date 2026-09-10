"""Join COMPLETE frozen judgments to all waterbodies; no inference or potential.

The campaign barrier precedes inventory access. Missing contexts, explicit null
draws, and numeric draws remain distinct; no neutral or adverse value is filled.
Aggregation is the existing pilot rule, not a new score calibration.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fpv_llm_v124 import ROOT, INVENTORY, TAGS, FIELDS
from src.analysis.prepare_global_scores_v124 import GLOBAL, sha, freeze_json, immutable_bytes
from src.analysis.run_global_subscription_v124 import MODELS, REPEATS, prompt_ordered_entity
from src.analysis.resume_global_scores_v124 import read_refs
from src.analysis.validate_global_campaign_v124 import validate
from src.analysis.evaluate_subscription_pilot_v124 import summarize_scores

SCORE_FIELDS = tuple((dimension, field) for dimension, fields in FIELDS.items() for field in fields)
SCORE_COLUMNS = tuple(field for _, field in SCORE_FIELDS)


def load_catalogue(path):
    # Frozen JSON is key-sorted; original prompt text uses insertion order.
    return [prompt_ordered_entity(e) for e in json.loads(Path(path).read_text())]


def assemble_inputs(mapping, entities, summaries, waterbody_ids):
    """Pure join with cardinality/range guards; callers prove raw completeness."""
    ids = pd.Index(waterbody_ids, name='wb_id')
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError('Empty, duplicate or missing inventory IDs')
    if mapping.wb_id.isna().any() or not mapping.wb_id.is_unique or set(mapping.wb_id) != set(ids):
        raise ValueError('Mapping does not cover the exact inventory')
    lookup = {e['entity_id']: e for e in entities}
    if len(lookup) != len(entities) or not entities:
        raise ValueError('Empty or duplicated entity catalogue')
    if any(e['dimension'] not in FIELDS for e in entities):
        raise ValueError('Unknown entity dimension')
    expected = {(e['entity_id'], e['dimension'], f) for e in entities for f in FIELDS[e['dimension']]}
    observed = list(summaries[['entity_id', 'dimension', 'field']].itertuples(index=False, name=None))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Incomplete, extra or duplicated entity field summaries')
    draws = len(MODELS) * REPEATS
    for row in summaries.itertuples():
        counts = [row.valid_observations, row.missing_observations, row.valid_providers, row.expected_observations]
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 0 for v in counts):
            raise ValueError('Invalid observation counts')
        if row.expected_observations != draws or row.valid_observations + row.missing_observations != draws:
            raise ValueError('Incomplete response draw counts')
        if not 0 <= row.valid_providers <= len(MODELS) or row.valid_providers > row.valid_observations:
            raise ValueError('Invalid provider coverage')
        if row.valid_observations > row.valid_providers * REPEATS:
            raise ValueError('Impossible numeric draw/provider coverage')
        lo, hi = FIELDS[row.dimension][row.field]
        if row.valid_observations == 0:
            if pd.notna(row.score) or row.valid_providers != 0:
                raise ValueError('Unknown judgments must remain null')
        elif (not isinstance(row.score, (int, float, np.integer, np.floating))
              or isinstance(row.score, (bool, np.bool_)) or not np.isfinite(row.score) or not lo <= row.score <= hi):
            raise ValueError('Invalid aggregate score')
    out = mapping.set_index('wb_id').loc[ids].reset_index()
    for dimension, field in SCORE_FIELDS:
        column = dimension + '_entity_id'
        known = out[column].dropna()
        valid_ids = {e['entity_id'] for e in entities if e['dimension'] == dimension}
        if not set(known) <= valid_ids:
            raise ValueError('Unknown or cross-dimension mapped entity: ' + dimension)
        table = summaries.loc[summaries.field.eq(field)].set_index('entity_id')
        out[field] = out[column].map(table.score).astype(float)
        out[field + '_context_missing'] = out[column].isna()
        for source, suffix in [('valid_observations', 'numeric_draws'), ('missing_observations', 'null_draws'),
                               ('valid_providers', 'numeric_providers')]:
            # A missing context was never submitted; it is not four null replies.
            out[field + '_' + suffix] = out[column].map(table[source]).fillna(0).astype(int)
    return out


def build_inputs(receipt_path):
    receipt_path = (ROOT / receipt_path).resolve()
    report = validate(receipt_path, require_complete=True)
    # Nothing below is reached for a partially scored campaign.
    receipt = json.loads(receipt_path.read_text())
    directory = receipt_path.parent.parent
    contract = json.loads((directory / 'contract.json').read_text())
    preparation_path = GLOBAL / 'preparation.json'
    preparation = json.loads(preparation_path.read_text())
    if not preparation['all_checks_passed'] or preparation['waterbodies'] != 199976 or preparation['entities'] != 4701:
        raise ValueError('Unexpected global preparation')
    for path in [INVENTORY, TAGS]:
        if sha(path) != preparation['source_hashes'][str(path.relative_to(ROOT))]:
            raise ValueError('Inventory or regional context source changed')
    for name in ['waterbody_entity_map.csv', 'entities.json']:
        if sha(GLOBAL / name) != preparation['artifact_hashes'][name]:
            raise ValueError('Frozen mapping or catalogue changed')
    entities = load_catalogue(GLOBAL / 'entities.json')
    lookup = {e['entity_id']: e for e in entities}
    records = list(read_refs(receipt['responses'], lookup, contract['environment']).values())
    observations, summaries = summarize_scores(records)
    mapping = pd.read_csv(GLOBAL / 'waterbody_entity_map.csv')
    inventory_ids = pd.read_parquet(INVENTORY, columns=['wb_id']).wb_id
    if len(inventory_ids) != preparation['waterbodies'] or len(entities) != preparation['entities']:
        raise ValueError('Actual inventory/catalogue counts differ from preparation')
    joined = assemble_inputs(mapping, entities, summaries, inventory_ids)
    destination = GLOBAL / 'score_inputs' / receipt_path.stem
    payload = BytesIO()
    joined.to_parquet(payload, index=False)
    immutable_bytes(destination / 'waterbody_scores.parquet', payload.getvalue())
    for name, table in [('observations', observations), ('entity_scores', summaries)]:
        immutable_bytes(destination / (name + '.csv'), table.to_csv(index=False, lineterminator='\n').encode())
    summary = dict(status='complete_score_inputs_not_potential_results', waterbodies=len(joined),
        entities=len(entities), successful_predictions=report['successful_predictions'],
        source_receipt=str(receipt_path.relative_to(ROOT)), source_receipt_sha256=sha(receipt_path),
        preparation_sha256=sha(preparation_path), code_sha256=sha(Path(__file__)),
        aggregation_code_sha256=sha(ROOT / 'src/analysis/evaluate_subscription_pilot_v124.py'),
        aggregation='mean within each provider, then equal mean across available providers',
        score_column_order=list(SCORE_COLUMNS), defaults_filled=False, global_potential_computed=False,
        scientific_accuracy_validated=False,
        output_hashes={p.name: sha(p) for p in destination.iterdir() if p.suffix in {'.csv', '.parquet'}},
        note='Fixed contemporary judgments; explicit null and unavailable context remain distinct. No future governance inference.')
    freeze_json(destination / 'validation.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    build_inputs(parser.parse_args().receipt)
