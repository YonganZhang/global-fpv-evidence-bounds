"""Offline reconstruction from published final answers; never invokes an SDK."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

def reconstruct(root,out):
    sys.path.insert(0,str(root/'historical_source'))
    from src.analysis.lean_global_consumers_v124 import assemble_primary_inputs
    from src.analysis.global_level_kernel_v124 import levels
    records=[]
    for name in ['primary','comparison']:
        records.extend(json.loads(s) for s in (root/'records'/f'{name}.jsonl').read_text().splitlines())
    mapping=pd.read_csv(root/'mapping/waterbody_entity_map.csv')
    entities=json.loads((root/'mapping/entities.json').read_text())
    plan=json.loads((root/'protocol/frozen_sampling_plan.json').read_text())
    scores,observations,primary,pairs=assemble_primary_inputs(mapping,entities,records,mapping.wb_id,plan['required_slots'])
    expected=pd.read_parquet(root/'reference/waterbody_scores.parquet')
    pd.testing.assert_frame_equal(scores,expected,check_dtype=False,check_exact=True)
    saved=pd.read_csv(root/'reference/audit_pairs.csv')
    keys=['entity_id','dimension','field']
    for field in ['codex_score','claude_score','absolute_delta']:
        a=pairs.set_index(keys)[field].sort_index();b=saved.set_index(keys)[field].sort_index()
        np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-15,equal_nan=True)
    ex=json.loads((root/'example/real_waterbody.json').read_text())
    fields=['ecological_suitability','maintenance_ease','renewable_support','finance_delivery','implementation_offset']
    row=scores.set_index('wb_id').loc[ex['wb_id']]
    for key in fields:assert row[key]==ex['scores'][key]
    # Dummy physical zeros are only a way to call the untouched kernel's mapping
    # functions. No zero generation is represented as an actual site outcome.
    mapped=levels([0.],[0.],np.array([False]),np.array([False]),[[ex['scores'][f] for f in fields]],strength=.5,missing_case='restricted').iloc[0]
    names=['ecological_area_fraction','development_area_fraction','engineering_cost_factor','maintenance_cost_factor']
    example=dict(wb_id=ex['wb_id'],scores=ex['scores'],mapping={name:float(mapped[name]) for name in names},interpretation='Weights and O&M multiplier only; physical inputs and site L1/L2/L3 not claimed.')
    g=pd.read_csv(root/'results/global_benefits.csv');g=g.loc[g.protection.eq('core') & g.missing_case.eq('restricted')].sort_values('endpoint')
    assert g.endpoint.tolist()==['lower','upper']
    headline=[]
    for _,r in g.iterrows():
        headline.append(dict(endpoint=r.endpoint,L1_TWh=r.L1_generation_twh,L2_TWh=r.L2_generation_twh,L3_TWh=r.L3_generation_twh,L2_L1_pct=100*r.L2_generation_twh/r.L1_generation_twh,L3_L1_pct=100*r.L3_generation_twh/r.L1_generation_twh,L3_demand_pct=r.L3_matched_generation_twh/r.matched_final_consumption_twh*100,L3_countries_ge_10pct_share=100*r.L3_countries_ge_10pct/r.countries_with_demand,L1_water_km3=r.L1_water_saving_known_subset_km3,L3_water_km3=r.L3_water_saving_known_subset_km3,L3_L1_water_pct=100*r.L3_water_saving_known_subset_km3/r.L1_water_saving_known_subset_km3,L3_L2_water_pct=100*r.L3_water_saving_known_subset_km3/r.L2_water_saving_known_subset_km3,water_equivalent_million_people=r.L3_water_saving_known_subset_km3*300/106))
    for field,expected_values in [('L2_L1_pct',[28.13,31.57]),('L3_L1_pct',[4.77,10.79]),('L3_demand_pct',[3.17,7.14]),('L3_L1_water_pct',[6.41,15.14]),('L3_L2_water_pct',[17.83,37.19]),('L3_water_km3',[14.03,33.15])]:
        assert [round(r[field],2) for r in headline]==expected_values,field
    future=pd.read_csv(root/'results/future_changes.csv')
    f=future.loc[future.protection.eq('core') & future.missing_case.eq('restricted') & future.ice_error_case.eq('central') & future.experiment.eq('ssp585') & future.period.eq('2081_2100')]
    assert len(f)==6
    changes={}
    for level,bounds in [('L1',[-3.55,-2.41]),('L2',[22.01,62.39]),('L3',[2.49,7.29])]:
        values=100*f[level+'_change_gwh']/f[level+'_historical_gwh']
        changes[level]=[float(values.min()),float(values.max())]
        assert [round(x,2) for x in changes[level]]==bounds
    out.mkdir(parents=True,exist_ok=True)
    scores.to_parquet(out/'waterbody_scores.parquet',index=False)
    observations.to_csv(out/'observations.csv',index=False)
    pairs.to_csv(out/'audit_pairs.csv',index=False)
    pd.DataFrame(headline).to_csv(out/'paper_headlines.csv',index=False)
    report=dict(status='passed',waterbodies=len(scores),primary_entities=len(primary.entity_id.unique()),comparison_entities=pairs.entity_id.nunique(),comparison_fields=len(pairs),score_table_matches_frozen_reference=True,example=example,future_ssp585_late_century_pct=changes,model_calls=0,scope='Exact final-answer to score mapping; weights and paper ratios rebuilt from frozen aggregate outputs. This does NOT rerun the restricted-data physical/geospatial pipeline.')
    (out/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reconstruct(a.root.resolve(),a.out.resolve())
