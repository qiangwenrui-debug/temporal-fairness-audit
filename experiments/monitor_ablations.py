#!/usr/bin/env python3
"""
Estimator ablations and policy checks on census data (monitoring protocol of delay_aware_monitor.py).

For each test year T (reference year H = latest labelled year, held out of training):
  Residual estimators (beta = 0)
    E0      last observed value (rule of year H on year H)
    E1, E2  soft labels, raw and group-wise isotonic
    E3u     current rule applied to H, all weights 1        <- ablation: rule update only
    E3      current rule applied to H, density-ratio weights clipped at the 99th percentile
    E3_c95, E3_c999, E3_noclip  same with other clipping levels
  Weight diagnostics: ESS share overall and within the (g, y = 1) cells, largest weight share.
  Approval-gap policies (beta interpolation of Eq. (1) and a fixed-budget variant)
    none, static, carry, per_period         as in the paper
    per_period_budget                        both thresholds shifted by a common constant so
                                             that exactly the share q is approved; beta* searched
                                             under that constraint
    pooled_retro                             beta*_pool computed retrospectively on all test years
  For every policy and year: overall approval rate, G, RG, FRR (qualified rejected),
  FAR (approval rate among y = 0), precision, and tie mass at the thresholds.

Usage (needs run_temporal_audit.py and delay_aware_monitor.py in the same folder):
    python monitor_ablations.py --states CA --task income --group race --label-mode quantile
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from run_temporal_audit import (TASKS, GROUP_COLS, load_real, add_group, build_model, coarsen,
                                thresholds, beta_star, BETAS)
from delay_aware_monitor import (rule_thresholds, apply_tau, residual, _domain_clf,
                                 isotonic_by_group, e3_parts)

warnings.filterwarnings("ignore")
CLIPS = {"E3_c95": 0.95, "E3": 0.99, "E3_c999": 0.999, "E3_noclip": None}


# ---------------------------------------------------------------- weights
def raw_ratio(XH, gH, XT, gT, seed=0, clf="lr"):
    """Cross-fitted p_T/p_H on the rows of H, unclipped (same folds as domain_weights)."""
    def design(X, g):
        X = np.asarray(X, float)
        return np.column_stack([X, g, X * g[:, None]]) if clf == "lr" else np.column_stack([X, g])
    ZH, ZT = design(XH, gH), design(XT, gT)
    rng = np.random.default_rng(seed)
    fH = rng.integers(0, 2, len(ZH)); fT = rng.integers(0, 2, len(ZT))
    p = np.empty(len(ZH))
    for k in (0, 1):
        Z = np.vstack([ZH[fH != k], ZT[fT != k]])
        lab = np.r_[np.zeros((fH != k).sum()), np.ones((fT != k).sum())]
        model = _domain_clf(clf, seed).fit(Z, lab)
        p[fH == k] = model.predict_proba(ZH[fH == k])[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return p / (1 - p) * (len(ZH) / len(ZT))


def clip_norm(w, q):
    w = w if q is None else np.minimum(w, np.quantile(w, q))
    return w / w.mean()


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w)) if len(w) else np.nan


def estimates(XH, sH, gH, yH, XT, sT, gT, q, seed=0, clf="lr", w_oracle=None):
    tauT = rule_thresholds(sT, gT, q, 0.0)
    aT = apply_tau(sT, gT, tauT).astype(float)
    aH_own = apply_tau(sH, gH, rule_thresholds(sH, gH, q, 0.0)).astype(float)
    aH_cur = apply_tau(sH, gH, tauT).astype(float)
    q1 = (yH == 1).astype(float)
    out = dict(E0=residual(aH_own, gH, q1), E1=residual(aT, gT, sT),
               E2=residual(aT, gT, isotonic_by_group(sH, gH, yH, sT, gT)),
               E3u=residual(aH_cur, gH, q1))
    w_raw = raw_ratio(XH, gH, XT, gT, seed, clf)
    for name, cq in CLIPS.items():
        w = clip_norm(w_raw, cq)
        out[name] = residual(aH_cur, gH, w * q1)
        if name == "E3":
            out["ess_frac"] = ess(w)
            for g in (0, 1):
                out[f"ess_g{g}y1"] = ess(w[(gH == g) & (yH == 1)])
            out["max_w_share"] = float(w.max() / w.sum())
            out.update(e3_parts(aH_cur, gH, yH, w))
        if name == "E3_noclip":
            out["ess_noclip"] = ess(w)
            out["ess_noclip_g1y1"] = ess(w[(gH == 1) & (yH == 1)])
    if w_oracle is not None:
        out["E3_oracle"] = residual(aH_cur, gH, w_oracle * q1)
    return out


# ---------------------------------------------------------------- fixed-budget thresholds
def _count(sorted_s, tau):
    return len(sorted_s) - np.searchsorted(sorted_s, tau, side="left")


def budget_taus(s, g, q, betas):
    """For each beta: tau_g(beta) + c with a common shift c chosen so that the number of
    approvals is as close as possible to round(q n) (bisection on c)."""
    betas = np.atleast_1d(betas)
    T, Tg = thresholds(s, g, q)
    s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
    t1 = (1 - betas) * T + betas * Tg[1]
    t0 = (1 - betas) * T + betas * Tg[0]
    k = int(round(q * len(s)))
    lo = np.full(len(betas), -1.0); hi = np.full(len(betas), 1.0)
    for _ in range(60):                      # count(c) is non-increasing in c
        c = 0.5 * (lo + hi)
        n = _count(s1, t1 + c) + _count(s0, t0 + c)
        too_many = n > k
        lo = np.where(too_many, c, lo); hi = np.where(too_many, hi, c)
    return t1 + hi, t0 + hi


def budget_gap_curve(s, g, q, betas):
    t1, t0 = budget_taus(s, g, q, betas)
    s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
    return _count(s1, t1) / len(s1) - _count(s0, t0) / len(s0)


def beta_star_budget(s, g, q, delta):
    G = budget_gap_curve(s, g, q, BETAS)
    ok = np.where(np.abs(G) <= delta)[0]
    return float(BETAS[ok[0]]) if len(ok) else np.nan


def policy_stats(s, g, y, tau):
    a = s >= np.where(g == 1, tau[1], tau[0])
    P1, P0 = a[g == 1].mean(), a[g == 0].mean()
    a11, a01 = a[(g == 1) & (y == 1)].mean(), a[(g == 0) & (y == 1)].mean()
    # tie mass: individuals whose score equals the smallest approved score of their group
    tie = 0
    for k in (0, 1):
        sk = s[(g == k) & a]
        if len(sk):
            tie += int((s[g == k] == sk.min()).sum())
    return dict(approval=a.mean(), G=P1 - P0, RG=a11 - a01, FRR=1 - a[y == 1].mean(),
                FAR=a[y == 0].mean(), precision=y[a].mean(), tie_share=tie / len(s),
                unique_share=len(np.unique(s)) / len(s))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="income", choices=list(TASKS))
    ap.add_argument("--group", default="race", choices=list(GROUP_COLS))
    ap.add_argument("--states", nargs="+", default=["CA"])
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    ap.add_argument("--label-delay", type=int, default=1)
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--data-dir", default="acs_data")
    ap.add_argument("--label-mode", default="quantile", choices=["nominal", "real", "quantile"])
    ap.add_argument("--base-year", type=int, default=2015)
    ap.add_argument("--label-quantile", type=float, default=0.6)
    ap.add_argument("--domain-clf", default="lr", choices=["hgb", "lr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.task}_{a.group}_{'-'.join(a.states)}_d{a.label_delay}"
    if a.task == "income":
        tag += f"_{a.label_mode}"
    out = a.out or f"ablations_{tag}"
    os.makedirs(out, exist_ok=True)

    data = load_real(a.task, a.states, a.years, a.data_dir, a.label_mode, a.base_year,
                     a.label_quantile)
    data = {y: add_group(d, a.group) for y, d in data.items()}
    years = sorted(data)
    spec = TASKS[a.task]
    feats = [f for f in spec["features"] if f != GROUP_COLS[a.group]]

    est_rows, pol_rows, cache = [], [], {}
    prev_b = prev_bb = static_b = static_bb = None
    for T in years:
        labelled = [t for t in years if t <= T - a.label_delay]
        if len(labelled) < 2:
            continue
        H = max(labelled)
        tr = pd.concat([data[t] for t in years if t < H], ignore_index=True)
        if len(tr) > a.max_train:
            tr = tr.sample(a.max_train, random_state=a.seed)
        model = build_model(a.model, feats, spec["categorical"], a.seed)
        model.fit(coarsen(tr[feats]), tr["__y"].values)
        dH, dT = data[H], data[T]
        XH, XT = coarsen(dH[feats]), coarsen(dT[feats])
        sH, sT = model.predict_proba(XH)[:, 1], model.predict_proba(XT)[:, 1]
        gH, yH, gT, yT = dH["__g"].values, dH["__y"].values, dT["__g"].values, dT["__y"].values
        cache[T] = (sT, gT, yT)

        est = estimates(XH.values, sH, gH, yH, XT.values, sT, gT, a.q, a.seed, a.domain_clf)
        truth = policy_stats(sT, gT, yT, rule_thresholds(sT, gT, a.q, 0.0))
        est_rows.append(dict(year=T, holdout=H, true_RG=truth["RG"], true_G=truth["G"],
                             n_H=len(sH), n_H_g1y1=int(((gH == 1) & (yH == 1)).sum()), **est))

        b = beta_star(sT, gT, a.q, a.delta); b = 0.0 if np.isnan(b) else b
        bb = beta_star_budget(sT, gT, a.q, a.delta); bb = 0.0 if np.isnan(bb) else bb
        if static_b is None:
            static_b, static_bb = b, bb
        interp = {"none": 0.0, "static": static_b, "carry": 0.0 if prev_b is None else prev_b,
                  "per_period": b}
        for pol, beta in interp.items():
            pol_rows.append(dict(year=T, policy=pol, beta=beta,
                                 **policy_stats(sT, gT, yT, rule_thresholds(sT, gT, a.q, beta))))
        for pol, beta in {"static_budget": static_bb,
                          "carry_budget": 0.0 if prev_bb is None else prev_bb,
                          "per_period_budget": bb}.items():
            t1, t0_ = budget_taus(sT, gT, a.q, beta)
            pol_rows.append(dict(year=T, policy=pol, beta=beta,
                                 **policy_stats(sT, gT, yT, {1: t1[0], 0: t0_[0]})))
        prev_b, prev_bb = b, bb
        e = est_rows[-1]
        print(f"  T={T} H={H} RG={e['true_RG']:+.4f} E0={e['E0']:+.4f} E3u={e['E3u']:+.4f} "
              f"E3={e['E3']:+.4f} E2={e['E2']:+.4f} beta*={b:.3f} beta*_budget={bb:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # retrospective pooled beta (same beta in every test year, pooled gap incl. composition)
    num1 = np.zeros_like(BETAS); num0 = np.zeros_like(BETAS); N1 = N0 = 0
    for T, (s, g, y) in cache.items():
        T_, Tg = thresholds(s, g, a.q)
        s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
        num1 += _count(s1, (1 - BETAS) * T_ + BETAS * Tg[1])
        num0 += _count(s0, (1 - BETAS) * T_ + BETAS * Tg[0])
        N1 += len(s1); N0 += len(s0)
    ok = np.where(np.abs(num1 / N1 - num0 / N0) <= a.delta)[0]
    bpool = float(BETAS[ok[0]]) if len(ok) else np.nan
    for T, (s, g, y) in cache.items():
        pol_rows.append(dict(year=T, policy="pooled_retro", beta=bpool,
                             **policy_stats(s, g, y, rule_thresholds(s, g, a.q, bpool))))

    est_df, pol_df = pd.DataFrame(est_rows), pd.DataFrame(pol_rows)
    est_df.to_csv(f"{out}/estimates.csv", index=False)
    pol_df.to_csv(f"{out}/policies.csv", index=False)

    E = ["E0", "E1", "E2", "E3u", "E3", "E3_c95", "E3_c999", "E3_noclip"]
    err = pd.DataFrame([dict(estimator=e,
                             MAE_pp=100 * (est_df[e] - est_df.true_RG).abs().mean(),
                             max_err_pp=100 * (est_df[e] - est_df.true_RG).abs().max(),
                             share_2021=(est_df.loc[est_df.year == 2021, e] /
                                         est_df.loc[est_df.year == 2021, "true_RG"]).mean()
                             if (est_df.year == 2021).any() else np.nan,
                             alerts=int((est_df[e].abs() > 0.02).sum()),
                             hits=int(((est_df[e].abs() > 0.02) & (est_df.true_RG.abs() > 0.02)).sum()))
                        for e in E])
    psum = pol_df.groupby("policy").agg(
        n_years=("G", "size"), years_outside=("G", lambda v: int((v.abs() > a.delta + 1e-12).sum())),
        max_abs_G_pp=("G", lambda v: 100 * v.abs().max()),
        approval_min=("approval", "min"), approval_max=("approval", "max"),
        FRR=("FRR", "mean"), FAR=("FAR", "mean"), precision=("precision", "mean"),
        max_abs_RG_pp=("RG", lambda v: 100 * v.abs().max()),
        tie_share_max=("tie_share", "max")).reset_index()
    with open(f"{out}/summary.md", "w") as f:
        f.write(f"# Monitoring ablations: {tag} ({a.model}, domain clf {a.domain_clf})\n\n")
        f.write(f"true residual years > 2 pp: {int((est_df.true_RG.abs() > 0.02).sum())} of "
                f"{len(est_df)}; retrospective pooled beta = {bpool}\n\n")
        f.write("## Estimator errors\n\n" + err.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Yearly estimates and weight diagnostics (pp)\n\n")
        show = est_df[["year", "holdout", "true_RG"] + E].copy()
        show[["true_RG"] + E] *= 100
        show = show.join(est_df[["ess_frac", "ess_g1y1", "ess_g0y1", "ess_noclip",
                                 "ess_noclip_g1y1", "max_w_share", "n_H_g1y1"]])
        f.write(show.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Policies (summary)\n\n" + psum.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Policies (yearly)\n\n" + pol_df.round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Runtime {time.time() - t0:.0f}s\n")
    print(open(f"{out}/summary.md").read()[:3000])


if __name__ == "__main__":
    main()
