#!/usr/bin/env python3
"""
Additional checks under the audit protocol of run_temporal_audit.py (train on years <= T - d).

For one setting and one model (default XGBoost) it reports, per test year:
  * G_t at beta = 0, beta*_t, and G_t when the pooled strength beta*_pool is applied in every
    year (does one beta keep several years within the tolerance at once?)
  * the overall approval rate at beta*_t and at beta*_pool (Eq. (1) does not fix it at q)
  * beta*_t under a fixed approval budget (both thresholds shifted by a common constant)
  * the pooled gap of Proposition 1 split into weighted mean and composition term
Usage:  python audit_checks.py --states CA --task income --group race --label-mode quantile
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from run_temporal_audit import (TASKS, GROUP_COLS, load_real, add_group, build_model, coarsen,
                                thresholds, beta_star, year_stats, BETAS)
from monitor_ablations import _count, beta_star_budget, budget_taus, policy_stats
from delay_aware_monitor import rule_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="income", choices=list(TASKS))
    ap.add_argument("--group", default="race", choices=list(GROUP_COLS))
    ap.add_argument("--states", nargs="+", default=["CA"])
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    ap.add_argument("--label-delay", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=["xgb"])
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--data-dir", default="acs_data")
    ap.add_argument("--label-mode", default="quantile", choices=["nominal", "real", "quantile"])
    ap.add_argument("--base-year", type=int, default=2015)
    ap.add_argument("--label-quantile", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.task}_{a.group}_{'-'.join(a.states)}_d{a.label_delay}"
    if a.task == "income":
        tag += f"_{a.label_mode}"
    out = a.out or f"revaudit_{tag}"
    os.makedirs(out, exist_ok=True)

    data = load_real(a.task, a.states, a.years, a.data_dir, a.label_mode, a.base_year,
                     a.label_quantile)
    data = {y: add_group(d, a.group) for y, d in data.items()}
    years = sorted(data)
    spec = TASKS[a.task]
    feats = [f for f in spec["features"] if f != GROUP_COLS[a.group]]
    test_years = [T for T in years if any(t <= T - a.label_delay for t in years)]

    rows, pooled_rows = [], []
    for m in a.models:
        sc = {}
        for T in test_years:
            tr = pd.concat([data[t] for t in years if t <= T - a.label_delay], ignore_index=True)
            if len(tr) > a.max_train:
                tr = tr.sample(a.max_train, random_state=a.seed)
            model = build_model(m, feats, spec["categorical"], a.seed)
            model.fit(coarsen(tr[feats]), tr["__y"].values)
            te = data[T]
            sc[T] = (model.predict_proba(coarsen(te[feats]))[:, 1], te["__g"].values, te["__y"].values)
            print(f"  {m} {T} [{time.time() - t0:.0f}s]", flush=True)
        # pooled beta* (same beta in every year; pooled gap including the composition term)
        num1 = np.zeros_like(BETAS); num0 = np.zeros_like(BETAS); N1 = N0 = 0
        for T, (s, g, y) in sc.items():
            T_, Tg = thresholds(s, g, a.q)
            s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
            num1 += _count(s1, (1 - BETAS) * T_ + BETAS * Tg[1])
            num0 += _count(s0, (1 - BETAS) * T_ + BETAS * Tg[0])
            N1 += len(s1); N0 += len(s0)
        ok = np.where(np.abs(num1 / N1 - num0 / N0) <= a.delta)[0]
        bpool = float(BETAS[ok[0]]) if len(ok) else np.nan
        for T, (s, g, y) in sc.items():
            b = beta_star(s, g, a.q, a.delta); b = 0.0 if np.isnan(b) else b
            bb = beta_star_budget(s, g, a.q, a.delta); bb = 0.0 if np.isnan(bb) else bb
            st0 = policy_stats(s, g, y, rule_thresholds(s, g, a.q, 0.0))
            stb = policy_stats(s, g, y, rule_thresholds(s, g, a.q, b))
            stp = policy_stats(s, g, y, rule_thresholds(s, g, a.q, bpool))
            t1, t0_ = budget_taus(s, g, a.q, bb)
            stbb = policy_stats(s, g, y, {1: t1[0], 0: t0_[0]})
            t1p, t0p = budget_taus(s, g, a.q, bpool)
            stpb = policy_stats(s, g, y, {1: t1p[0], 0: t0p[0]})
            ys = year_stats(s, g, y, a.q)
            rows.append(dict(model=m, year=T, n=len(s), n1=int(g.sum()),
                             G0=st0["G"], RG0=st0["RG"], RC0=ys["RC"], approval0=st0["approval"],
                             beta_star=b, G_bstar=stb["G"], approval_bstar=stb["approval"],
                             FRR_bstar=stb["FRR"], FAR_bstar=stb["FAR"], FRR0=st0["FRR"], FAR0=st0["FAR"],
                             beta_pool=bpool, G_bpool=stp["G"], approval_bpool=stp["approval"],
                             RG_bpool=stp["RG"], beta_star_budget=bb, G_bstar_budget=stbb["G"],
                             approval_bstar_budget=stbb["approval"], G_bpool_budget=stpb["G"],
                             unique_share=st0["unique_share"], tie_share=st0["tie_share"],
                             P1=ys["P1"], P0=ys["P0"]))
        d = pd.DataFrame([r for r in rows if r["model"] == m])
        w1 = d.n1 / d.n1.sum(); w0 = (d.n - d.n1) / (d.n - d.n1).sum()
        wmean, comp = (w1 * d.G0).sum(), ((w1 - w0) * d.P0).sum()
        pooled_rows.append(dict(model=m, pooled_gap=(w1 * d.P1).sum() - (w0 * d.P0).sum(),
                                weighted_mean=wmean, composition=comp, beta_pool=bpool,
                                min_G=d.G0.min(), max_G=d.G0.max(),
                                years_within_at_bpool=int((d.G_bpool.abs() <= a.delta + 1e-12).sum()),
                                years_within_at_0=int((d.G0.abs() <= a.delta + 1e-12).sum()),
                                n_years=len(d), max_abs_approval_dev_bstar=(d.approval_bstar - a.q).abs().max(),
                                max_abs_approval_dev_bpool=(d.approval_bpool - a.q).abs().max(),
                                max_abs_beta_diff_budget=(d.beta_star - d.beta_star_budget).abs().max()))
    df, pdf = pd.DataFrame(rows), pd.DataFrame(pooled_rows)
    df.to_csv(f"{out}/yearly.csv", index=False)
    pdf.to_csv(f"{out}/pooled.csv", index=False)
    with open(f"{out}/summary.md", "w") as f:
        f.write(f"# Audit checks: {tag}\n\n## Pooled\n\n" + pdf.round(4).to_markdown(index=False))
        show = df.copy()
        for c in [c for c in df.columns if c.startswith(("G", "RG", "RC"))]:
            show[c] = 100 * show[c]
        f.write("\n\n## Yearly (gaps in pp)\n\n" + show.round(4).to_markdown(index=False))
        f.write(f"\n\nRuntime {time.time() - t0:.0f}s\n")
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
