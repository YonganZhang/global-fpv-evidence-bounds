"""Independent structural, provenance and publication gate. No inference or network."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import sys
import pandas as pd

def validate(root):
    check=[]
    def require(name,yes):
        if not yes:raise ValueError(name)
        check.append(name)
    manifest=json.loads((root/'checksums.json').read_text())
    for rel,digest in manifest.items():require('hash '+rel,hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest)
    sys.path.insert(0,str(root/'historical_source'))
    from src.analysis.fpv_llm_v124 import validate_response,build_entity
    from src.analysis.run_subscription_fpv_scores_v124 import parse_model_json,schema_for
    from src.analysis.recover_lean_scores_v124 import eligible
    entities=json.loads((root/'mapping/entities.json').read_text());lookup={x['entity_id']:x for x in entities}
    records=[];rejected=[]
    for name,n in [('primary',4701),('comparison',471),('rejected_attempts',18)]:
        rows=[json.loads(x) for x in (root/'records'/f'{name}.jsonl').read_text().splitlines()]
        require(name+' count',len(rows)==n)
        (rejected if name=='rejected_attempts' else records).extend(rows)
    required={(x['entity_id'],x['provider'],x['repeat']) for x in json.loads((root/'protocol/frozen_sampling_plan.json').read_text())['required_slots']}
    require('exact frozen membership',{(x['entity_id'],x['provider'],x['repeat']) for x in records}==required)
    dims={'ecology':'ecology','engineering':'maintenance','country':'national','province':'provincial'}
    for r in records:
        require('valid final JSON '+r['signature'],validate_response(r['dimension'],parse_model_json(r['response']['final_response']))==r['scores'])
        e=lookup[r['entity_id']];require('context '+r['signature'],build_entity(r['dimension'],r['context'])==e)
        folder=root/'protocol'/dims[r['dimension']]
        prompt=(folder/'user_template.txt').read_text().replace('{{context_json}}',json.dumps(r['context'],ensure_ascii=False))
        require('exact request '+r['signature'],r['job']['prompt']==prompt and r['job']['policy']==(folder/'system.txt').read_text() and r['job']['schema']==schema_for(r['dimension']))
        require('no event stream '+r['signature'],'raw_events' not in r['response'])
        require('single turn without active tools '+r['signature'],r['response']['turns']==1 and not r['response']['active_tool_items'])
        require('unspecified decoding retained '+r['signature'],not {'temperature','top_p','seed','max_tokens','max_output_tokens'} & set(r['job']))
    histories=collections.defaultdict(list)
    for r in sorted(rejected+records,key=lambda r:r['generated_at']):
        slot=(r['entity_id'],r['provider'],r['repeat'])
        require('retry allowed '+r['signature'],eligible(histories[slot]));histories[slot].append(r)
    require('all final attempts accepted',all(rs[-1]['status']=='ok' for rs in histories.values()))
    mapping=pd.read_csv(root/'mapping/waterbody_entity_map.csv')
    require('minimal mapping allowlist',list(mapping)==['wb_id','ecology_entity_id','engineering_entity_id','country_entity_id','province_entity_id'])
    require('complete mapping',len(mapping)==199976 and mapping.wb_id.is_unique)
    # Scan publishable text for concrete secret formats, not harmless variable names
    # in the frozen credential-cleaning code. Never print matching strings.
    patterns=[r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'\bsk-[A-Za-z0-9_-]{30,}',r'Bearer\s+[A-Za-z0-9._-]{30,}']
    for rel in manifest:
        p=root/rel
        if p.suffix in {'.json','.jsonl','.md','.txt','.py','.csv','.yml','.cff'}:
            content=p.read_text()
            require('no secret-like value '+rel,not any(re.search(x,content) for x in patterns))
    require('no provider raw GIS files',not any(p.suffix.lower() in {'.shp','.dbf','.tif','.tiff','.gpkg'} for p in root.rglob('*')))
    return dict(status='passed',checks=len(check),accepted_primary=4701,accepted_comparison=471,rejected_attempts=18,max_attempts=max(map(len,histories.values())),hashes_checked=len(manifest),no_internal_reasoning_exported=True,no_new_model_calls=True,limits='Credential format scan plus exact field allowlists is not independent scientific validation or a legal opinion.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--report',type=Path,required=True);a=p.parse_args();result=validate(a.root.resolve());a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
