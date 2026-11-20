"""Paired household intervals for the 2019-to-2021 origin and race-definition replays."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from acs_group_flags import ROOT, STATES
from model_cohort_replay import BootstrapScores, sample_weights
from origin_replay import population, score_batch, group_definitions, origin_components
from policy_monitor_data import FIPS


def prepare(root,state,year,fit,flags_manifest):
    people = population(root,state,year,fit.target_raw_sha256,flags_manifest)
    score = score_batch(root,state,year,people,fit)
    hispanic = people.HISP.to_numpy()!=1
    definitions = group_definitions(people.RAC1P.to_numpy(dtype=int),
                                    people.RACWHT.to_numpy(dtype=bool),
                                    people.RACBLK.to_numpy(dtype=bool),hispanic)
    union = np.logical_or.reduce([mask for mask, g in definitions.values()])
    household = people.SERIALNO[union]
    if household.isna().any():
        raise ValueError(f"Missing household identifier for {state} {year}")
    codes = pd.factorize(household,sort=False)[0]
    prepared = {}
    for name,(mask,group) in definitions.items():
        s = score[mask]
        g = group[mask].astype(int)
        h = hispanic[mask].astype(int)
        prepared[name] = (BootstrapScores(s,g),s,g,h,mask[union])
    return codes,prepared


def audit(batch,counts,beta):
    result = {}
    for name,(orders,s,g,h,select) in batch.items():
        w = counts[select]
        thresholds,gap,_ = orders.thresholds_and_gap(w,np.array([beta]))
        if name == "alone":
            cell = 2*g+h
            n = np.bincount(cell,weights=w,minlength=4).reshape(2,2)
            passed = s>=thresholds[0,g]
            a = np.bincount(cell[passed],weights=w[passed],minlength=4).reshape(2,2)
            calculated = 100*np.diff(a.sum(axis=1)/n.sum(axis=1))[0]
            np.testing.assert_allclose(calculated,gap[0],rtol=0,atol=1e-11)
        else:
            n,a = None,None
        result[name] = (gap[0],n,a,thresholds[0])
    return result


def paired_metrics(prior,current):
    metrics = {}
    primary_change = current["alone"][0]-prior["alone"][0]
    for name in current:
        old,new = prior[name][0],current[name][0]
        metrics[name,"prior_gap_pp"] = old
        metrics[name,"gap_pp"] = new
        metrics[name,"change_pp"] = new-old
        if name != "alone":
            metrics[name,"prior_minus_primary_pp"] = old-prior["alone"][0]
            metrics[name,"target_minus_primary_pp"] = new-current["alone"][0]
            metrics[name,"change_minus_primary_pp"] = new-old-primary_change
    mix,within,standard = origin_components(prior["alone"],current["alone"])
    metrics["alone","origin_mix_pp"] = mix
    metrics["alone","within_origin_pp"] = within
    metrics["alone","prior_mix_at_target_cutoffs_pp"] = standard
    return metrics


def intervals(root,state,draws):
    out = root/"results/diagnostics"
    fits = pd.read_csv(root/"results/transfer/extended_fit_manifest.csv").query("state==@state").set_index("target_year")
    proj = pd.read_csv(root/"data/group_reporting/manifest.csv").query("state==@state").set_index("year")
    observed = pd.read_csv(out/"origin_replay.csv",float_precision="round_trip")
    observed = observed.query("state==@state and year==2021 and policy=='common'").set_index("definition")
    if len(observed)!=3 or observed.beta.nunique()!=1:
        raise ValueError("Expected all three group definitions at a common strength")
    beta = observed.beta.iloc[0]
    prior_h,prior_batch = prepare(root,state,2019,fits.loc[2019],proj.loc[2019])
    target_h,target_batch = prepare(root,state,2021,fits.loc[2021],proj.loc[2021])
    old = audit(prior_batch,np.ones(len(prior_h),dtype=int),beta)
    now = audit(target_batch,np.ones(len(target_h),dtype=int),beta)
    point = paired_metrics(old,now)
    for (name,metric),value in point.items():
        if metric in observed:
            np.testing.assert_allclose(value,observed.loc[name,metric],rtol=0,atol=1e-10)
    seed = 90700 + FIPS[state]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,len(point)))
    for k in range(draws):
        old = audit(prior_batch,sample_weights(prior_h,rng),beta)
        now = audit(target_batch,sample_weights(target_h,rng),beta)
        samples[k] = list(paired_metrics(old,now).values())
    lower,upper = np.quantile(samples,[.025,.975],axis=0)
    return [dict(state=state,year=2021,reference_year=2019,policy="common",beta=beta,
                 definition=name,metric=metric,estimate_pp=value,lower_pp=lower[i],upper_pp=upper[i],
                 prior_union_n=len(prior_h),target_union_n=len(target_h),
                 prior_union_households=int(prior_h.max()+1),target_union_households=int(target_h.max()+1),
                 draws=draws,seed=seed) for i,((name,metric),value) in enumerate(point.items())]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--draws",type=int,default=1000)
    args=parser.parse_args()
    if args.draws<2:
        parser.error("--draws must be at least two")
    root=args.root.resolve()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs=[executor.submit(intervals,root,s,args.draws) for s in STATES]
        rows=[row for job in jobs for row in job.result()]
    path=root/"results/diagnostics/origin_intervals.csv"
    temp=path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temp,index=False)
    temp.replace(path)
