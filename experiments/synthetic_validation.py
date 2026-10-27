#!/usr/bin/env python3
"""
Synthetic validation with known ground truth (method component + decomposition).

Four drift scenarios over 9 periods; group 1 is the minority (20%):
  none      : nothing changes (control)
  covariate : group-1 feature means drift downwards; P(y | x) fixed      (covariate shift)
  label     : group-1 base rate falls; P(x | y) fixed                     (label / prior shift)
  concept   : P(y | x, g=1) changes (group-1 intercept falls); P(x) fixed (concept shift)

For each period t >= 2 a model is trained on periods < t-1, period t-1 is the labelled
holdout H (label delay 1), and period t is scored. Reports, averaged over replicates:
  * trajectories of G_t, RG_t, RC_t (which component moves under which shift)
  * error of the residual estimators E0-E3 against the realised RG_t
Expected: E3 is accurate under covariate shift; every label-free estimator is biased
under label and concept shift, which marks the boundary of the method.

Usage: python synthetic_validation.py [--reps 20] [--n 20000]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from run_temporal_audit import year_stats
from delay_aware_monitor import estimate_all

D, PERIODS, SHARE1 = 5, 9, 0.2
BETA = np.array([1.0, 0.8, -0.6, 0.5, 0.3])
EST = ["E0", "E1", "E2", "E3", "E3_oracle"]


def gen(scn, t, n, rng):
    g = (rng.random(n) < SHARE1).astype(int)
    k = t / (PERIODS - 1)                       # 0 -> 1 over the horizon
    if scn == "label":
        pi = np.where(g == 1, 0.40 - 0.20 * k, 0.40)
        y = (rng.random(n) < pi).astype(int)
        mu = np.where(y[:, None] == 1, 0.5, -0.5) * np.ones(D)
        X = mu + rng.normal(size=(n, D))
        return X, g, y
    mu = np.zeros((n, D))
    mu[g == 1, :2] = -0.3                       # baseline group difference
    if scn == "covariate":
        mu[g == 1, :2] -= 0.8 * k
    X = mu + rng.normal(size=(n, D))
    logit = X @ BETA - 0.5
    if scn == "concept":
        logit = logit - 1.2 * k * (g == 1)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, g, y


def oracle_ratio(scn, X, g, tH, tT):
    """True density ratio p_T(x | g) / p_H(x | g) (group shares are constant)."""
    k = lambda t: t / (PERIODS - 1)
    w = np.ones(len(g))
    if scn == "covariate":
        mH, mT = -0.3 - 0.8 * k(tH), -0.3 - 0.8 * k(tT)
        z = X[:, :2]
        lr = np.exp(-((z - mT) ** 2).sum(1) / 2 + ((z - mH) ** 2).sum(1) / 2)
        w = np.where(g == 1, lr, 1.0)
    elif scn == "label":
        f1 = np.exp(-((X - 0.5) ** 2).sum(1) / 2); f0 = np.exp(-((X + 0.5) ** 2).sum(1) / 2)
        pi = lambda t: 0.40 - 0.20 * k(t)
        lr = (pi(tT) * f1 + (1 - pi(tT)) * f0) / (pi(tH) * f1 + (1 - pi(tH)) * f0)
        w = np.where(g == 1, lr, 1.0)
    return w


def run(scn, n, reps, q, seed0):
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        data = [gen(scn, t, n, rng) for t in range(PERIODS)]
        for t in range(2, PERIODS):
            Xtr = np.vstack([data[s][0] for s in range(t - 1)])
            ytr = np.concatenate([data[s][2] for s in range(t - 1)])
            m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)   # group-blind model
            XH, gH, yH = data[t - 1]
            XT, gT, yT = data[t]
            sH, sT = m.predict_proba(XH)[:, 1], m.predict_proba(XT)[:, 1]
            st = year_stats(sT, gT, yT, q)
            est = estimate_all(XH, sH, gH, yH, XT, sT, gT, q, 0.0, 0.0, seed=r, clf="lr",
                               w_oracle=oracle_ratio(scn, XH, gH, t - 1, t))
            rows.append(dict(scenario=scn, rep=r, t=t, G=st["G"], RG=st["RG"], RC=st["RC"],
                             **{e: est[e] for e in EST}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--out", default="synthetic_validation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.concat([run(s, a.n, a.reps, a.q, 1000 * i)
                    for i, s in enumerate(["none", "covariate", "label", "concept"])])
    df.to_csv(f"{a.out}/synthetic_runs.csv", index=False)

    traj = df.groupby(["scenario", "t"])[["G", "RG", "RC"]].mean().mul(100).round(2).reset_index()
    for e in EST:
        df[f"err_{e}"] = 100 * (df[e] - df.RG).abs()
        df[f"bias_{e}"] = 100 * (df[e] - df.RG)
    err = df.groupby("scenario")[[f"err_{e}" for e in EST]].mean().round(2)
    bias = df.groupby("scenario")[[f"bias_{e}" for e in EST]].mean().round(2)
    first_last = traj[traj.t.isin([2, PERIODS - 1])]

    with open(f"{a.out}/synthetic_summary.md", "w") as f:
        f.write(f"# Synthetic validation (reps = {a.reps}, n = {a.n} per period, q = {a.q})\n\n")
        f.write("## Which component moves (mean over reps, pp; first vs last test period)\n\n"
                + first_last.to_markdown(index=False) + "\n\n")
        f.write("## Residual estimation: mean absolute error (pp)\n\n" + err.to_markdown() + "\n\n")
        f.write("## Residual estimation: mean signed bias (pp)\n\n" + bias.to_markdown() + "\n\n")
        f.write("## Full trajectories (pp)\n\n" + traj.to_markdown(index=False) + "\n")
    print(open(f"{a.out}/synthetic_summary.md").read())


if __name__ == "__main__":
    main()
