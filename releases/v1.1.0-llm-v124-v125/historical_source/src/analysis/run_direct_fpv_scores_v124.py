"""Bounded, resumable direct-judgment pilot. Never logs credentials.

Provider route verified against https://doc.zhizengzeng.com/doc-3979947.
Requests carry public geographic context only; no manuscript or private files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import requests

from src.analysis.fpv_llm_v124 import OUT, ROOT, FIELDS, validate_response

BASE = 'https://api.zhizengzeng.com/v1'
HELPER = '/mnt/data/yongan-admin-2/.codex/skills/share-docs/scripts/get-credential.sh'
PROMPT_VERSION = 'fpv-separated-direct-v124-01'
SYSTEM = '''You assess uncertain regional conditions for a scientific floating-PV scenario.
Use your pretrained knowledge directly; do not browse, invent citations, or claim to have
read a policy or visited a site. No particular country is assigned a target score.
Provide judgments, not observed facts or calibrated deployment probabilities. Return
only a JSON object with exactly the requested fields. Each field contains score,
confidence (low/medium/high/unknown), and a concise reason (maximum 60 words).
If you cannot make even a regional judgment, use score:null and confidence:unknown.
Do not manufacture site-specific species, permits, costs, dates or infrastructure.
The physical pipeline already handles irradiation, temperature, ice duration,
protected-area membership, historical drying, depth, roads, grid distance and
population proximity. Do NOT rescore those measured quantities or hard exclusions.
Only assess the residual factor specified below. Distinguish assumptions from knowledge.
For scores on [0,1], 0 means strongly adverse, .25 adverse, .5 mixed, .75 favorable,
1 strongly favorable for the specified residual factor, NOT feasible area percentages.
Your scores will be transformed by researcher-declared scenario mappings, not treated
as empirical observations. A reservoir is not automatically ecologically harmless.
'''
INSTRUCTIONS = {
    'ecology': '''Output ecological_suitability in [0,1]: residual ecological compatibility
of limited floating-PV coverage for this waterbody type/ecoregional context, considering
light-dependent freshwater habitat and aquatic ecosystem sensitivity. Terrestrial
ecoregion/biome is a coarse contextual proxy, NOT a freshwater ecological survey.
Do not infer protected status, local water uses, population, evaporative benefit,
climate/ice or engineering costs. Natural lakes and reservoirs must be distinguished.
State the important missing freshwater information in the reason.''',
    'engineering': '''Output maintenance_ease in [0,1]: residual ease of operation and
maintenance of freshwater floating-PV equipment under this regional climate context,
limited to corrosion/humidity and biological fouling/cleaning requirements. This is
NOT a dollar estimate. Do not rescore PV yield, temperature efficiency, ice eligibility,
depth/anchoring, waves, flooding, road or grid distance, policy or ecology. Water salinity
and local water chemistry are unknown: do not invent them. State uncertainty.''',
    'country': '''Output renewable_support and finance_delivery, each in [0,1].
renewable_support: general institutional support and implementation continuity for
renewable energy applicable to FPV, not a claim of site permission or a specific law.
finance_delivery: relative investment financing and project-delivery conditions,
not measured capex, discount rates, physical grid access, market demand or irradiance.
Assess the stated current snapshot, not future policies. If recent developments are
unknown, acknowledge that. Give a separate rationale for each factor.''',
    'province': '''Output implementation_offset in [-0.15,0.15]: residual provincial
implementation support relative to the country's general renewable-energy conditions.
Zero means no inferred relative difference; positive favorable, negative adverse.
Do not reuse national conditions, grid/population distances, demand, climate or costs.
If provincial knowledge is unavailable return null, not zero. No future policy forecast.''',
}


def build_request(entity, model):
    dimension = entity['dimension']
    example = {k: dict(score=None, confidence='unknown', reason='Explain the judgment and uncertainty.')
               for k in FIELDS[dimension]}
    user = (INSTRUCTIONS[dimension] + '\nContext: ' + json.dumps(entity['context'], ensure_ascii=False)
            + '\nRequired structure: ' + json.dumps(example))
    request = dict(model=model, messages=[dict(role='system', content=SYSTEM), dict(role='user', content=user)],
                   temperature=.2, max_tokens=900, response_format={'type': 'json_object'})
    return request


def load_credential():
    key = os.environ.get('ZHIZENGZENG_API_KEY', '').strip()
    if not key:
        result = subprocess.run([HELPER, 'ZHIZENGZENG_API_KEY'], capture_output=True, text=True, timeout=30)
        key = result.stdout.strip() if result.returncode == 0 else ''
    if not key:
        raise RuntimeError('Registered credential unavailable; no key material logged')
    return key


def run_call(entity, model, repeat, key, directory):
    payload = build_request(entity, model)
    signature = hashlib.sha256(json.dumps(dict(request=payload, repeat=repeat, prompt_version=PROMPT_VERSION),
                                         sort_keys=True).encode()).hexdigest()
    path = directory / (signature + '.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('signature') != signature:
            raise ValueError('Cache signature mismatch')
        # Failures are kept too. Reattempts require an explicit new output run.
        return saved
    record = dict(signature=signature, entity_id=entity['entity_id'], dimension=entity['dimension'],
        context=entity['context'], requested_model=model, repeat=repeat, prompt_version=PROMPT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(), request=payload, provider=BASE,
        source_mode='direct_pretrained_judgment_no_retrieval', status='failed')
    try:
        response = requests.post(BASE + '/chat/completions', headers={'Authorization': 'Bearer ' + key},
                                 json=payload, timeout=(15, 90))
        record['http_status'] = response.status_code
        if response.status_code != 200:
            record['error_kind'] = 'http_failure'
            try:
                error_type = response.json().get('error', {}).get('type')
            except (ValueError, AttributeError):
                error_type = None
            # Persist only known categories, never arbitrary provider error text.
            if error_type in {'quota_not_enough', 'insufficient_quota', 'invalid_api_key', 'rate_limit_error'}:
                record['provider_error_type'] = error_type
        else:
            raw = response.json()
            content = raw.get('choices', [{}])[0].get('message', {}).get('content')
            record.update(returned_model=raw.get('model'), response_id=raw.get('id'),
                          usage=raw.get('usage'), response_content=content)
            result = json.loads(content)
            record['scores'] = validate_response(entity['dimension'], result)
            record['status'] = 'ok'
    except Exception as exc:
        # No exception text, HTTP headers or request auth are persisted.
        record['error_kind'] = type(exc).__name__
    # Guard against a malicious/unexpected service echoing the bearer credential.
    encoded = json.dumps(record, ensure_ascii=False, indent=2)
    if key in encoded:
        raise RuntimeError('Credential echo detected; response not saved')
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded + '\n')
    return record


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt-4.1-mini', 'qwen3-max'])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-calls', type=int, default=64)
    parser.add_argument('--run-id', default='pilot01')
    args = parser.parse_args()
    if not args.run_id.isalnum() or not 1 <= args.workers <= 4 or not 1 <= args.repeats <= 4:
        raise ValueError('Invalid bounded-run arguments')
    if len(set(args.models)) != len(args.models):
        raise ValueError('Repeated model identifier')
    entities_path = OUT / 'data/pilot_entities.json'
    entities = json.loads(entities_path.read_text())
    jobs = [(entity, model, repeat) for entity in entities for model in args.models for repeat in range(args.repeats)]
    if len(jobs) > args.max_calls or args.max_calls > 128:
        raise ValueError('Pilot call limit exceeded')
    directory = OUT / 'raw_direct' / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    key = load_credential()
    # One real task request first: quota/auth failures must not fan out globally.
    records = [run_call(*jobs[0], key, directory)]
    preflight_passed = records[0]['status'] == 'ok'
    if preflight_passed:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_call, *job, key, directory) for job in jobs[1:]]
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                print(len(records), '/', len(jobs), row['dimension'], row['requested_model'], row['status'], flush=True)
    records.sort(key=lambda r: (r['entity_id'], r['requested_model'], r['repeat']))
    report = dict(status=('pilot_calls_completed_not_scientifically_validated' if preflight_passed
                          else 'blocked_preflight_no_new_scores'), planned_calls=len(jobs),
        attempted_or_cached_calls=len(records), preflight_passed=preflight_passed,
        preflight_error=records[0].get('provider_error_type', records[0].get('error_kind')),
        successful_calls=sum(r['status'] == 'ok' for r in records), models=args.models, repeats=args.repeats,
        entities_sha256=hashlib.sha256(entities_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        responses=[dict(path=str((directory / (r['signature'] + '.json')).relative_to(ROOT)),
                        sha256=hashlib.sha256((directory / (r['signature'] + '.json')).read_bytes()).hexdigest())
                   for r in records])
    (OUT / 'reports' / (args.run_id + '_calls.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'responses'}, indent=2))
    if not preflight_passed or report['successful_calls'] != len(jobs):
        raise SystemExit(2)


if __name__ == '__main__':
    run_main()
