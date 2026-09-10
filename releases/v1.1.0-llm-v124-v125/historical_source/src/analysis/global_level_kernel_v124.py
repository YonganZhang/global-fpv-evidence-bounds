"""Vectorized separated L2/L3 kernel, shared by current and future inputs.

Scores are scenario judgments, not measured available area or costs. This module
does not fetch scores or infer that a partial campaign is globally complete.
"""
import numpy as np
import pandas as pd


def flag(frame, column):
    values=frame[column]
    if values.isna().any() or not values.isin([True,False]).all():
        raise ValueError('Undefined/nonboolean gate: '+column)
    return values.to_numpy(bool)


def masks(frame, endpoint, protection='core', ice_days=None):
    if endpoint not in {'lower','upper'} or protection not in {'core','conservative'}:
        raise ValueError('Invalid gate scenario')
    area=flag(frame,'l1_area_pass_v112')
    protected=flag(frame,'in_wdpa_polygon_strict_i_iv') if protection=='core' else (
        flag(frame,'in_wdpa_polygon_any')|flag(frame,'in_ramsar_polygon'))
    if ice_days is None:
        ice=flag(frame,'ice_pass_v112')
    else:
        days=np.asarray(ice_days,dtype=float)
        if days.shape!=(len(frame),) or np.isinf(days).any() or ((days[np.isfinite(days)]<0)|(days[np.isfinite(days)]>365)).any():
            raise ValueError('Invalid future ice array')
        ice=np.isfinite(days)&(days<=182.5) if endpoint=='lower' else ~np.isfinite(days)|(days<=182.5)
    for column in ['historical_dryup_evidence_known_v113','historical_no_dryup_observed_1991_2018_v113']:
        values=frame[column].dropna()
        if not values.isin([True,False]).all(): raise ValueError('Nonboolean drying history: '+column)
    known=frame.historical_dryup_evidence_known_v113.fillna(False).to_numpy(bool)
    dry_field=frame.historical_no_dryup_observed_1991_2018_v113
    if (known & dry_field.isna().to_numpy()).any(): raise ValueError('Known drying history has no result')
    no_dry=dry_field.fillna(False).to_numpy(bool)
    dry=known&no_dry if endpoint=='lower' else ~known|no_dry
    road='road_within_10km_'+('guaranteed' if endpoint=='lower' else 'possible')+'_v115'
    grid='grid_within_25km_'+('observed' if endpoint=='lower' else 'possible')+'_or_hydropower_v115'
    return area&~protected&ice&dry, flag(frame,road)&flag(frame,grid)&flag(frame,'within_10km_population_center')


def levels(l1, cost, env, access, scores, strength=.5, missing_case='unadjusted'):
    l1,cost=np.asarray(l1,float),np.asarray(cost,float)
    env,access=np.asarray(env),np.asarray(access)
    scores=np.asarray(scores,float)
    n=len(l1)
    if l1.shape!=(n,) or cost.shape!=(n,) or scores.shape!=(n,5) or env.shape!=(n,) or access.shape!=(n,):
        raise ValueError('Unaligned vector inputs')
    if env.dtype!=bool or access.dtype!=bool: raise ValueError('Gates must be boolean')
    if not np.isfinite(l1).all() or not np.isfinite(cost).all() or (l1<0).any() or (cost<0).any():
        raise ValueError('Invalid physical input')
    if not 0<=strength<=1 or missing_case not in {'unadjusted','restricted'} or np.isinf(scores).any():
        raise ValueError('Invalid scenario or scores')
    if ((scores[:,:4]<0)|(scores[:,:4]>1)).any() or ((scores[:,4]<-.15)|(scores[:,4]>.15)).any():
        raise ValueError('Score outside elicited range')
    missing=np.isnan(scores)
    default=np.array([1.,1.,1.,1.,0.] if missing_case=='unadjusted' else [0.,0.,0.,0.,-.15])
    values=np.where(missing,default,scores)
    eco=1-strength*(1-values[:,0])
    development=1-strength*(1-np.clip((values[:,2]+values[:,3])/2+values[:,4],0,1))
    maintenance=1+strength*(1-values[:,1])
    om_pv=.02*sum(1/1.08**year for year in range(1,26))
    factor=1+om_pv/(1+om_pv)*(maintenance-1)
    adjusted=cost*factor
    l2=l1*env*eco
    l3=l2*access*(adjusted<=100)*development
    return pd.DataFrame(dict(L1_gwh=l1,L2_gwh=l2,L3_gwh=l3,adjusted_cost_usd_mwh=adjusted,
        ecological_area_fraction=eco,development_area_fraction=development,
        engineering_cost_factor=factor,maintenance_cost_factor=maintenance,missing_components=missing.sum(axis=1)))
