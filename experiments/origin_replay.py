"""Audit the roles of recorded Hispanic origin and race-group membership in threshold transfer."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import folktables
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from acs_group_flags import ROOT, STATES, YEARS
from run_temporal_audit import RELSHIPP_TO_RELP, TASKS, coarsen


def population(root, state, year, digest, flags_manifest):
    directory = root / ("data/early_rolling" if year < 2014 else "data")
    name = f"{state}_{year}.parquet"
    raw = pd.read_parquet(directory / "normalized" / name)
    path = root / "data/group_reporting" / name
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != flags_manifest.projection_sha256:
            raise ValueError(f"Unmatched origin projection for {state} {year}")
    if digest != flags_manifest.zip_sha256 or len(raw) != flags_manifest.rows:
        raise ValueError(f"Inconsistent source for {state} {year}")
    flags = pd.read_parquet(path)
    if len(flags) != len(raw):
        raise ValueError(f"Wrong projection length for {state} {year}")
    raw = raw.join(flags,validate="one_to_one")
    if "RELP" not in raw:
        raw["RELP"] = raw.RELSHIPP.map(RELSHIPP_TO_RELP)
    return folktables.ACSIncome._preprocess(raw)


def score_batch(root,state,year,frame,fit):
    cache = root / "data/early_rolling"
    model_id = f"{state}_income_race_quantile_s0"
    model = XGBClassifier(n_jobs=3)
    model.load_model(cache / "models" / model_id / f"{fit.training_cutoff}.ubj")
    features = [f for f in TASKS["income"]["features"] if f!="RAC1P"]
    if features != fit.features.split("|") or fit.training_cutoff >= year:
        raise ValueError(f"Wrong feature order or cutoff: {state} {year}")
    scores = model.predict_proba(coarsen(frame[features].fillna(-1)))[:,1].astype(float)
    with np.load(cache / f"scores/audit/{state}_income_race_quantile_d1_s0/{year}.npz") as archived:
        primary = frame.RAC1P.isin([1,2])
        np.testing.assert_array_equal(scores[primary],archived["s"])
        np.testing.assert_array_equal((frame.loc[primary,"RAC1P"]==2).to_numpy(),archived["g"])
    return scores


def group_definitions(race,white,black,hispanic):
    primary = (race==1) | (race==2)
    return {"alone":(primary,race==2),
            "alone_nonhispanic":(primary & ~hispanic,race==2),
            "including_combinations":(white != black,black)}


def threshold_replay(scores, groups, hispanic, beta):
    pooled = np.quantile(scores,.25)
    counts = np.empty((2,2),dtype=int)
    approvals = np.empty_like(counts)
    thresholds = []
    for g in (0,1):
        select = groups==g
        s,h = scores[select],hispanic[select]
        threshold = (1-beta)*pooled + beta*np.quantile(s,.25)
        thresholds.append(threshold)
        counts[g] = np.bincount(h,minlength=2)
        approvals[g] = np.bincount(h[s>=threshold],minlength=2)
    rates = approvals.sum(axis=1)/counts.sum(axis=1)
    return 100*(rates[1]-rates[0]), counts, approvals, thresholds


def origin_components(prior, current):
    _,old_n,old_a,_ = prior
    _,new_n,new_a,_ = current
    if min(old_n.min(),new_n.min()) == 0:
        raise ValueError("Empty race-by-origin cell in the primary audit")
    old_share = old_n / old_n.sum(axis=1, keepdims=True)
    new_share = new_n / new_n.sum(axis=1, keepdims=True)
    old_rate,new_rate = old_a/old_n,new_a/new_n
    share_change = ((new_share-old_share)*(new_rate+old_rate)/2).sum(axis=1)
    rate_change = ((new_rate-old_rate)*(new_share+old_share)/2).sum(axis=1)
    mix = 100*(share_change[1]-share_change[0])
    within = 100*(rate_change[1]-rate_change[0])
    standardized = (old_share*new_rate).sum(axis=1)
    np.testing.assert_allclose(current[0]-prior[0],mix+within,atol=1e-10)
    return mix,within,100*(standardized[1]-standardized[0])


def replay_state(root,state):
    base = root / "results/transfer"
    fits = pd.read_csv(base/"extended_fit_manifest.csv").query("state==@state").set_index("target_year")
    fields = pd.read_csv(root/"data/group_reporting/manifest.csv").query("state==@state").set_index("year")
    policies = pd.concat([pd.read_csv(base/"from2010"/f) for f in ("rolling_state_year.csv","rolling_previous.csv")])
    policies = policies[policies.state==state]
    composition,contrasts,cells = [],[],[]
    previous_batch = None
    for i,year in enumerate(YEARS):
        source = fields.loc[year]
        fit = fits.loc[year] if year in fits.index else None
        frame = population(root,state,year,fit.target_raw_sha256 if fit is not None else source.zip_sha256,source)
        race = frame.RAC1P.to_numpy(dtype=int)
        hispanic = frame.HISP.to_numpy()!=1
        white = frame.RACWHT.to_numpy(dtype=bool)
        black = frame.RACBLK.to_numpy(dtype=bool)
        for r in range(1,10):
            for h in (0,1):
                n = np.sum((race==r)&(hispanic==h))
                composition.append(dict(state=state,year=year,rac1p=r,hispanic=h,n=n,total_eligible=len(frame)))
        if fit is None:
            continue
        scores = score_batch(root,state,year,frame,fit)
        batch = {}
        for group_def,(mask,g) in group_definitions(race,white,black,hispanic).items():
            batch[group_def] = (scores[mask],g[mask],hispanic[mask].astype(int))
        if previous_batch is not None:
            for policy in policies[policies.year==year].itertuples():
                for definition,data in batch.items():
                    old = threshold_replay(*previous_batch[definition],policy.beta)
                    new = threshold_replay(*data,policy.beta)
                    gap,counts,approvals,tau = new
                    common = dict(state=state,year=year,reference_year=YEARS[i-1],policy=policy.policy,
                                  definition=definition,beta=policy.beta)
                    row = dict(**common,n_white=counts[0].sum(),n_black=counts[1].sum(),
                               white_tau=tau[0],black_tau=tau[1],prior_gap_pp=old[0],
                               gap_pp=gap,change_pp=gap-old[0])
                    if definition=="alone":
                        np.testing.assert_allclose([gap,*tau],[policy.gap_pp,policy.tau0,policy.tau1],atol=1e-10)
                        np.testing.assert_array_equal(approvals.sum(axis=1),[policy.approved0,policy.approved1])
                        mix,within,standard = origin_components(old,new)
                        row.update(origin_mix_pp=mix,within_origin_pp=within,prior_mix_at_target_cutoffs_pp=standard)
                    contrasts.append(row)
                    for g in (0,1):
                        for h in (0,1):
                            if definition=="alone_nonhispanic" and h==1:
                                continue
                            cells.append(dict(**common,black=g,hispanic=h,n=counts[g,h],approved=approvals[g,h],
                                              prior_n=old[1][g,h],prior_approved=old[2][g,h]))
        previous_batch = batch
    return composition,contrasts,cells


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    root=args.root.resolve()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs=[executor.submit(replay_state,root,state) for state in STATES]
        result=[job.result() for job in jobs]
    out=root/"results/diagnostics"
    out.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(("eligible_race_origin","origin_replay","origin_cells")):
        table=pd.DataFrame([row for state in result for row in state[i]])
        table.to_csv(out/(name+".csv"),index=False)
        print(name,len(table))
