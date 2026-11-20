"""Separate the change in the scored cohort from updating the fitted model in the ACS rolling audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from extended_audit_scores import EARLY, LATER, ROOT, STATES, file_hash
from policy_monitor import panel
from policy_monitor_data import FIPS
from transfer_checks import household_codes


POLICIES = ("pooled", "previous", "common")
Q = .75


def cutoffs(scores, groups, strengths):
    pooled = np.quantile(scores, 1-Q)
    group_q = np.array([np.quantile(scores[groups == g], 1-Q) for g in (0,1)])
    return (1-strengths[:,None])*pooled + strengths[:,None]*group_q


def rates(scores, groups, thresholds):
    counts = np.array([[(scores[groups==g] >= t[g]).sum() for g in (0,1)] for t in thresholds])
    rate = 100 * (counts / np.bincount(groups))
    return rate[:,1]-rate[:,0], counts


def sample_weights(households, rng):
    n = households.max()+1
    repeats = np.bincount(rng.integers(n, size=n),minlength=n)
    return repeats[households]


class BootstrapScores:
    """Reusable orders for household-weighted empirical quantiles and approval counts."""
    def __init__(self, s, g):
        self.orders = [np.argsort(s)]
        for group in (0,1):
            members = np.flatnonzero(g==group)
            self.orders.append(members[np.argsort(s[members])])
        self.sorted = [s[order] for order in self.orders]

    def thresholds_and_gap(self, weights, strengths):
        cumulative = [np.r_[0, np.cumsum(weights[order])] for order in self.orders]
        quantiles = []
        for prefix, values in zip(cumulative,self.sorted):
            rank = (prefix[-1]-1)*(1-Q)
            ranks = [int(np.floor(rank)),int(np.ceil(rank))]
            vals = values[np.searchsorted(prefix[1:],ranks,side="right")]
            quantiles.append(vals[0]+(rank-ranks[0])*(vals[1]-vals[0]))
        thresholds = (1-strengths[:,None])*quantiles[0] + strengths[:,None]*np.array(quantiles[1:])
        return thresholds, self.gap(cumulative,thresholds), cumulative

    def gap(self, cumulative, thresholds):
        rate = []
        for group in (0,1):
            prefix = cumulative[group+1]
            below = np.searchsorted(self.sorted[group+1],thresholds[:,group],side="left")
            rate.append(100*((prefix[-1]-prefix[below])/prefix[-1]))
        return rate[1]-rate[0]


def paired_intervals(old_ref, old_t, new_t, ref_g, target_g, ref_h, target_h, strengths, draws, seed):
    ref = BootstrapScores(old_ref,ref_g)
    before = BootstrapScores(old_t,target_g)
    after = BootstrapScores(new_t,target_g)
    rng = np.random.default_rng(seed)
    results = np.empty((draws,len(strengths),5))
    for k in range(draws):
        weights_ref = sample_weights(ref_h,rng)
        weights_t = sample_weights(target_h,rng)
        threshold_r, gap_r, _ = ref.thresholds_and_gap(weights_ref,strengths)
        _, gap_t_old, prefix = before.thresholds_and_gap(weights_t,strengths)
        gap_t_fixed = before.gap(prefix,threshold_r)
        _, gap_t_new, _ = after.thresholds_and_gap(weights_t,strengths)
        results[k] = np.column_stack((gap_t_old-gap_r,gap_t_new-gap_t_old,
                                      gap_t_fixed-gap_r,gap_t_old-gap_t_fixed,gap_t_old))
        if k == 0:
            # Cross-check weighted orders against the ordinary person-level expanded draw.
            rr = np.repeat(np.arange(len(ref_g)),weights_ref)
            tt = np.repeat(np.arange(len(target_g)),weights_t)
            for s,g,ids,est in ((old_ref,ref_g,rr,gap_r),(old_t,target_g,tt,gap_t_old),(new_t,target_g,tt,gap_t_new)):
                th = cutoffs(s[ids],g[ids],strengths)
                np.testing.assert_allclose(rates(s[ids],g[ids],th)[0],est,rtol=0,atol=1e-11)
    return np.quantile(results,[.025,.975],axis=0)


def evaluate(root,state,draws,threads):
    dest = root / "results/transfer/from2010"
    policies = pd.concat([pd.read_csv(dest/p,float_precision="round_trip") for p in
                          ("rolling_state_year.csv","rolling_previous.csv")])
    policies = policies[policies.state==state].set_index(["year","policy"]).sort_index()
    manifest = pd.read_csv(root/"results/transfer/extended_fit_manifest.csv")
    manifest = manifest[manifest.state==state].set_index("target_year")
    data = root/"data/early_rolling"
    cache = data/"scores/audit"/f"{state}_income_race_quantile_d1_s0"
    model_dir = data/"models"/f"{state}_income_race_quantile_s0"
    issued = {}
    rows, intervals = [], []
    previous_model = None
    years = EARLY + LATER
    with threadpool_limits(limits=threads):
        for pos,year in enumerate(years[1:],1):
            last = years[pos-1]
            info = manifest.loc[year]
            source = (data if year < 2014 else root/"data")
            samples,features,_ = panel(source,state,[year],"income","race","quantile")
            cohort = samples[year]
            group = cohort["__g"].to_numpy(dtype=np.int64)
            path = model_dir/f"{last}.ubj"
            score_file = cache/f"{year}.npz"
            assert int(info.training_cutoff)==last < year and list(info.features.split("|"))==features
            assert file_hash(path)==info.model_sha256 and file_hash(score_file)==info.scores_sha256
            model = XGBClassifier(n_jobs=threads)
            model.load_model(path)
            score = model.predict_proba(cohort[features])[:,1].astype(float)
            with np.load(score_file) as z:
                np.testing.assert_array_equal(z["s"],score)
                np.testing.assert_array_equal(z["g"],group)
                np.testing.assert_array_equal(z["y"],cohort["__y"])
                hh = household_codes(source/f"normalized/{state}_{year}.parquet",group,z["y"])
            issued[year] = (score,group,hh)
            if previous_model is None:
                previous_model = model
                continue
            previous_score,previous_group,previous_hh = issued[last]
            old_score = previous_model.predict_proba(cohort[features])[:,1].astype(float)
            assert years[pos-2] < last < year
            block = policies.loc[year].loc[list(POLICIES)]
            beta = block.beta.to_numpy()
            ref_cuts = cutoffs(previous_score,previous_group,beta)
            ref_gap,_ = rates(previous_score,previous_group,ref_cuts)
            fixed_gap,_ = rates(old_score,group,ref_cuts)
            old_cuts = cutoffs(old_score,group,beta)
            old_gap,_ = rates(old_score,group,old_cuts)
            current_cuts = cutoffs(score,group,beta)
            current_gap,current_counts = rates(score,group,current_cuts)
            np.testing.assert_array_equal(current_counts,block[["approved0","approved1"]])
            np.testing.assert_allclose(current_cuts,block[["tau0","tau1"]],rtol=0,atol=1e-14)
            np.testing.assert_allclose(current_gap,block.gap_pp,rtol=0,atol=1e-11)
            np.testing.assert_allclose(old_gap-ref_gap,(fixed_gap-ref_gap)+(old_gap-fixed_gap),atol=1e-11)
            delta = np.column_stack((old_gap-ref_gap,current_gap-old_gap,fixed_gap-ref_gap,old_gap-fixed_gap,old_gap))
            ci_names = ("cohort_change_pp","model_change_pp","fixed_cut_cohort_change_pp",
                        "rethreshold_change_pp","old_model_target_gap_pp")
            have_prior = last in policies.index.get_level_values("year")
            for j,policy in enumerate(POLICIES):
                prior_issued = policies.loc[(last,policy),"gap_pp"] if have_prior else np.nan
                prior_beta = policies.loc[(last,policy),"beta"] if have_prior else np.nan
                row = dict(state=state,year=year,policy=policy,beta=beta[j],reference_year=last,
                           old_model_through=years[pos-2],new_model_through=last,
                           prior_issued_beta=prior_beta,prior_issued_gap_pp=prior_issued,
                           strength_change_pp=ref_gap[j]-prior_issued,prior_gap_pp=ref_gap[j],
                           old_threshold_target_gap_pp=fixed_gap[j],target_gap_pp=current_gap[j],
                           reference_tau_white=ref_cuts[j,0],reference_tau_black=ref_cuts[j,1],
                           old_model_target_tau_white=old_cuts[j,0],old_model_target_tau_black=old_cuts[j,1],
                           issued_tau_white=current_cuts[j,0],issued_tau_black=current_cuts[j,1],
                           reference_n=len(previous_group),target_n=len(group))
                row.update(dict(zip(ci_names,delta[j])))
                if have_prior:
                    assert abs((current_gap[j]-prior_issued) -
                               (row["strength_change_pp"]+delta[j,0]+delta[j,1])) < 1e-11
                rows.append(row)
            if draws:
                seed = 40500 + 100*(year-2010) + FIPS[state]
                lo,hi = paired_intervals(previous_score,old_score,score,previous_group,group,
                                         previous_hh,hh,beta,draws,seed)
                for j,policy in enumerate(POLICIES):
                    for m,name in enumerate(ci_names):
                        intervals.append(dict(state=state,year=year,policy=policy,metric=name,
                                              estimate=delta[j,m],lower_pp=lo[j,m],upper_pp=hi[j,m],
                                              draws=draws,seed=seed))
            previous_model = model
    return rows,intervals


def main(root,draws,workers,threads):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(evaluate,root,state,draws,threads) for state in STATES]
        results = [j.result() for j in jobs]
    out = root/"results/transfer/from2010"
    for i,name in enumerate(("model_cohort_replay","model_cohort_intervals")):
        rows = [row for result in results for row in result[i]]
        if rows:
            target = out/f"{name}.csv"
            tmp = target.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(tmp,index=False)
            tmp.replace(target)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--draws",type=int,default=1000,help="Paired household draws per state-year")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    if args.draws < 0 or args.draws==1:
        parser.error("--draws must be zero or at least two")
    main(args.root.resolve(),args.draws,args.workers,args.threads)
