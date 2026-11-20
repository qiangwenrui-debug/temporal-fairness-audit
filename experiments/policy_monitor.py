"""Walk-forward policy-conditional monitoring on official ACS person records. See --help."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import folktables
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import xgboost

from run_temporal_audit import TASKS, GROUP_COLS, RELSHIPP_TO_RELP, CPI_U, income_label, add_group, coarsen, build_model, beta_star
from monitor_ablations import beta_star_budget, budget_taus, ess
from delay_aware_monitor import rule_thresholds, apply_tau, residual, domain_weights, isotonic_by_group, e3_parts, gamma_needed, sensitivity_interval
from policy_monitor_data import YEARS

RULES = ("none", "per_period", "per_period_budget")
ALERT = 0.02


def panel(root, state, years, task, group, label):
    problem = {"income": folktables.ACSIncome, "employment": folktables.ACSEmployment}[task]
    cols = set(problem.features) | {"RELSHIPP", "PWGTP", problem.target}
    if task == "income":
        cols.add("ADJINC")
    feats = [f for f in TASKS[task]["features"] if f != GROUP_COLS[group]]
    data = {}
    years_meta = []
    for year in years:
        src = root / "normalized" / f"{state}_{year}.parquet"
        meta = json.loads(src.with_suffix(".json").read_text(encoding="utf8"))
        raw = pd.read_parquet(src, columns=[c for c in meta["columns"] if c in cols])
        if "RELP" not in raw:
            raw["RELP"] = raw["RELSHIPP"].map(RELSHIPP_TO_RELP)
        ref = problem._preprocess(raw)
        if task == "income":
            if label == "quantile" and year not in CPI_U:
                # One positive common annual CPI factor cancels from a within-year quantile.
                adj = ref.PINCP * ref.ADJINC / 1e6
                labels = adj > np.quantile(adj, .6)
            else:
                labels = income_label(ref, year, label, 2015, .6)
        else:
            labels = problem.target_transform(ref[problem.target])
        df = ref[problem.features].copy().fillna(-1)
        df["__y"] = np.asarray(labels, dtype=np.uint8)
        df = add_group(df, group)
        df[feats] = coarsen(df[feats])
        years_meta.append({"year":year, "raw_rows":meta["raw_person_records"], "filtered_before_group":len(ref),
                           "analysed":len(df), "g1":int(df.__g.sum()), "zip_sha256":meta["sha256"]})
        data[year] = df
    return data, feats, years_meta


def save_scores(directory, year, s, g, y):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{year}.npz"
    if not final.exists():
        temp = directory / f"{year}.{os.getpid()}.tmp.npz"
        np.savez_compressed(temp, s=np.asarray(s, dtype=np.float32), g=np.asarray(g, dtype=np.uint8), y=np.asarray(y, dtype=np.uint8))
        temp.replace(final)


def fitted(root, config, data, feats, groups, seed, last, threads):
    path = root / "models" / config / f"{last}.ubj"
    path.parent.mkdir(parents=True, exist_ok=True)
    m = build_model("xgb", feats, TASKS[groups]["categorical"], seed)
    m.set_params(n_jobs=threads)
    if path.exists():
        m.load_model(path)
        return m
    train = pd.concat([data[y] for y in sorted(data) if y <= last], ignore_index=True)
    if len(train) > 400000:
        train = train.sample(400000, random_state=seed)
    m.fit(train[feats], train["__y"].values)
    tmp = path.with_name(f"{last}.{os.getpid()}.ubj")
    m.save_model(tmp)
    tmp.replace(path)
    return m


def interval_action(lo, hi):
    if lo > ALERT or hi < -ALERT:
        return "outside"
    if lo >= -ALERT and hi <= ALERT:
        return "inside"
    return "abstain"


def simulate(args):
    label = args.label if args.task == "income" else "employed"
    years = YEARS + ([2024] if args.extend else [])
    data, feats, years_meta = panel(args.data, args.state, years, args.task, args.group, label)
    model_id = f"{args.state}_{args.task}_{args.group}_{label}_s{args.seed}"
    set_id = f"{args.state}_{args.task}_{args.group}_{label}_d{args.delay}_s{args.seed}"
    dest = args.out / set_id
    dest.mkdir(parents=True, exist_ok=True)
    info = {"state":args.state,"task":args.task,"group":args.group,"label":label,"delay":args.delay,"seed":args.seed,"q":args.q}
    manifest = {**info,"years":years_meta,"approval_q":args.q,"gap_tolerance":.05,"residual_tolerance":ALERT,
                "prediction_dtype":"float32","training_cap":400000,"xgboost":xgboost.__version__,"sklearn":sklearn.__version__,
                "numpy":np.__version__,"pandas":pd.__version__,"python":sys.version,"threads":args.threads,
                "input_feature_order":feats,"policy_grid_points":1001}
    (dest/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    # A model trained through K suffices for any score set using that expanding window.
    # Seed, task, group, state, label and K fully identify the fit and the training sample.
    audit_dir = args.data / "scores" / "audit" / set_id
    monitor_dir = args.data / "scores" / "monitor" / set_id
    history = {r:{} for r in RULES}
    rows, intervals = [], []
    for T in years:
        eligible = [u for u in years if u <= T - args.delay]
        if not eligible:
            continue
        H = max(eligible)
        target = data[T]
        xT = target[feats]
        gT, yT = target.__g.values, target.__y.values
        if not (audit_dir/f"{T}.npz").exists():
            am = fitted(args.data, model_id, data, feats, args.task, args.seed, H, args.threads)
            save_scores(audit_dir, T, am.predict_proba(xT)[:, 1], gT, yT)
        prev = [u for u in years if u < H]
        if not prev:
            continue
        m = fitted(args.data, model_id, data, feats, args.task, args.seed, max(prev), args.threads)
        reference = data[H]
        xH = reference[feats]
        gH, yH = reference.__g.values, reference.__y.values
        sH, sT = m.predict_proba(xH)[:, 1], m.predict_proba(xT)[:, 1]
        save_scores(monitor_dir, T, sT, gT, yT)
        soft = isotonic_by_group(sH, gH, yH, sT, gT)
        w = domain_weights(xH, gH, xT, gT, seed=args.seed, clip_q=.99, clf="lr")
        b = beta_star(sT, gT, args.q, .05)
        bb = beta_star_budget(sT, gT, args.q, .05)
        if not np.isfinite(b) or not np.isfinite(bb):
            raise ValueError(f"No feasible grid setting for {set_id} {T}")
        t1, t0 = budget_taus(sT, gT, args.q, bb)
        rules = [(RULES[0], 0., rule_thresholds(sT, gT, args.q, 0.)), (RULES[1], b, rule_thresholds(sT, gT, args.q, b)),
                 (RULES[2], bb, {1:float(t1[0]), 0:float(t0[0])})]
        cut = "external_2024" if T == 2024 else "original_years"
        for rule, beta, tau in rules:
            at = apply_tau(sT, gT, tau).astype(float)
            ah = apply_tau(sH, gH, tau).astype(float)
            # All estimates only access reference labels and current-period scores/covariates.
            # E0 is the realized historical residual under the actual deployed rule of H.
            h = history[rule]
            e0 = h[H]["truth"] if H in h else np.nan
            est = {"E0":e0, "E1":residual(at,gT,sT),"E2":residual(at,gT,soft),
                   "E3u":residual(ah,gH,yH),"E3":residual(ah,gH,w*yH)}
            parts = e3_parts(ah,gH,yH,w)
            past = [rec["gamma"] for yr,rec in h.items() if yr<=T-args.delay]
            cal = max(past, default=np.nan)
            # Values below create only retrospective validation fields, never inputs to estimates.
            truth = residual(at,gT,yT)
            gap = at[gT==1].mean()-at[gT==0].mean()
            gneed = gamma_needed(parts, truth)
            h[T] = {"truth":truth, "gamma":gneed}
            shared = dict(**info, window=cut, year=T, holdout=H, train_through=max(prev), policy=rule, beta=beta)
            records = dict(**shared,approval=at.mean(), true_G=gap,true_RG=truth,FRR=1-at[yT==1].mean(),FAR=at[yT==0].mean(),
                           tau0=tau[0],tau1=tau[1],nT=len(gT),nH=len(gH),ess=ess(w),ess_g1y1=ess(w[(gH==1)&(yH==1)]),
                           gamma_needed=gneed, gamma_history=cal, history_k=len(past),**parts,**est)
            if rule != "none" and abs(gap)>.05+1e-10:
                raise AssertionError(f"Selected infeasible policy {shared}: {gap}")
            for name, value in est.items():
                records[name+"_ae_pp"]=100*abs(value-truth)
                records[name+"_alarm"]=(abs(value)>ALERT) if np.isfinite(value) else np.nan
            rows.append(records)
            for name, gamma in [("history",cal),("fixed_1.1",1.1),("fixed_1.25",1.25),("fixed_1.5",1.5)]:
                if not np.isfinite(gamma):
                    continue
                lo, hi = sensitivity_interval(parts, gamma)
                action = interval_action(lo, hi)
                actual = "outside" if abs(truth)>ALERT else "inside"
                intervals.append(dict(**shared,gamma_method=name,gamma=gamma,history_k=len(past),lo_pp=100*lo,hi_pp=100*hi,
                                      width_pp=100*(hi-lo), true_RG_pp=100*truth,covered=lo<=truth<=hi,action=action,
                                      actual=actual,incorrect_definite=(action!="abstain" and action!=actual)))
        print(f"computed {set_id} {T} H={H} n={len(target)}",flush=True)
    for file, data_rows in [("yearly.csv",rows),("intervals.csv",intervals)]:
        temp = dest/(file+".tmp")
        pd.DataFrame(data_rows).to_csv(temp,index=False)
        temp.replace(dest/file)
    print(f"complete {set_id}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--state",choices=["CA","TX","FL","NY"],required=True)
    parser.add_argument("--task",choices=["income","employment"],default="income")
    parser.add_argument("--group",choices=["race","sex"],default="race")
    parser.add_argument("--label",choices=["quantile","real","nominal"],default="quantile")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--q",type=float,default=.75,help="approval budget for the operating-point sensitivity")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--delay",type=int,default=1)
    parser.add_argument("--extend",action="store_true",help="append 2024 temporal validation after rolling through original years")
    options=parser.parse_args()
    with threadpool_limits(limits=options.threads):
        simulate(options)
