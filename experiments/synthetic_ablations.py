#!/usr/bin/env python3
"""
Synthetic ablations (known ground truth; logistic models only, deterministic).

Part 1  Ablation and clipping in the four scenarios of synthetic_validation.py:
        E3u (current rule on the reference period, weights 1) and E3 with clipping at the
        95th / 99th / 99.9th percentile or none.
Part 2  Label shift (base rates diverge) with group-specific P(x | y, g), under three rules:
          retrained  model retrained each period, thresholds = per-period quantiles
          fixed_rule one model (trained on period 0) and fixed absolute thresholds
          fixed_model_requantiled  the fixed model with per-period quantile thresholds
        If the score function and thresholds are fixed and P(x | y, g) is stable, a_gy and
        hence RG_t are constant (Corollary 1); a rule that is re-estimated can move RG_t.
Part 3  When does reweighting pay? Covariate shift of strength m in {0.4, 0.8, 1.6} (sd units
        over the horizon), reference sizes n_H in {2,000; 5,000; 20,000}, and a well-specified
        or misspecified scoring model (true logit has a quadratic term the model lacks).

Usage: python synthetic_ablations.py [--reps 20] [--n 20000]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from run_temporal_audit import year_stats, thresholds
from synthetic_validation import gen, oracle_ratio, D, PERIODS, SHARE1, BETA
from monitor_ablations import estimates

EST = ["E0", "E1", "E2", "E3u", "E3", "E3_c95", "E3_c999", "E3_noclip", "E3_oracle"]


# ---------------------------------------------------------------- part 1
def part1(n, reps):
    rows = []
    for i, scn in enumerate(["none", "covariate", "label", "concept"]):
        for r in range(reps):
            rng = np.random.default_rng(1000 * i + r)
            data = [gen(scn, t, n, rng) for t in range(PERIODS)]
            for t in range(2, PERIODS):
                Xtr = np.vstack([data[s][0] for s in range(t - 1)])
                ytr = np.concatenate([data[s][2] for s in range(t - 1)])
                m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                XH, gH, yH = data[t - 1]; XT, gT, yT = data[t]
                sH, sT = m.predict_proba(XH)[:, 1], m.predict_proba(XT)[:, 1]
                st = year_stats(sT, gT, yT, 0.75)
                e = estimates(XH, sH, gH, yH, XT, sT, gT, 0.75, seed=r, clf="lr",
                              w_oracle=oracle_ratio(scn, XH, gH, t - 1, t))
                rows.append(dict(scenario=scn, rep=r, t=t, RG=st["RG"], **{k: e[k] for k in EST},
                                 ess=e["ess_frac"], ess_g1y1=e["ess_g1y1"], ess_noclip=e["ess_noclip"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part 2
def gen_label_g(t, n, rng):
    """Label shift: base rate of group 1 falls 0.40 -> 0.20 and that of group 0 rises
    0.40 -> 0.60; P(x | y, g) is fixed but group-specific (group 1 sits lower)."""
    g = (rng.random(n) < SHARE1).astype(int)
    k = t / (PERIODS - 1)
    pi = np.where(g == 1, 0.40 - 0.20 * k, 0.40 + 0.20 * k)
    y = (rng.random(n) < pi).astype(int)
    mu = np.where(y[:, None] == 1, 0.5, -0.5) * np.ones(D)
    mu[:, :2] -= 0.8 * g[:, None]            # group-1 class-conditionals sit lower
    return mu + rng.normal(size=(n, D)), g, y


def part2(n, reps, q=0.75):
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(5000 + r)
        data = [gen_label_g(t, n, rng) for t in range(PERIODS)]
        m_fix = LogisticRegression(max_iter=1000).fit(data[0][0], data[0][2])
        s1 = m_fix.predict_proba(data[1][0])[:, 1]
        T_fix, _ = thresholds(s1, data[1][1], q)
        for t in range(2, PERIODS):
            X, g, y = data[t]
            Xtr = np.vstack([data[s][0] for s in range(t - 1)])
            ytr = np.concatenate([data[s][2] for s in range(t - 1)])
            m_rt = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            s_rt, s_fx = m_rt.predict_proba(X)[:, 1], m_fix.predict_proba(X)[:, 1]
            for rule, s, tau in (("retrained", s_rt, None), ("fixed_model_requantiled", s_fx, None),
                                 ("fixed_rule", s_fx, T_fix)):
                if tau is None:
                    st = year_stats(s, g, y, q)
                else:
                    a = s >= tau
                    a11, a10 = a[(g == 1) & (y == 1)].mean(), a[(g == 1) & (y == 0)].mean()
                    a01, a00 = a[(g == 0) & (y == 1)].mean(), a[(g == 0) & (y == 0)].mean()
                    pi1, pi0 = y[g == 1].mean(), y[g == 0].mean()
                    st = dict(G=a[g == 1].mean() - a[g == 0].mean(), RG=a11 - a01,
                              RC=(1 - pi1) * (a10 - a11) - (1 - pi0) * (a00 - a01))
                    st["approval"] = a.mean()
                rows.append(dict(rep=r, t=t, rule=rule, G=st["G"], RG=st["RG"], RC=st["RC"],
                                 approval=st.get("approval", q)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part 3
def gen_cov(t, n, rng, m, spec):
    g = (rng.random(n) < SHARE1).astype(int)
    k = t / (PERIODS - 1)
    mu = np.zeros((n, D))
    mu[g == 1, :2] = -0.3 - m * k
    X = mu + rng.normal(size=(n, D))
    logit = X @ BETA - 0.5
    if spec == "misspecified":
        logit = logit + 0.6 * (X[:, 0] ** 2 - 1)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, g, y


def oracle_cov(X, g, tH, tT, m):
    k = lambda t: t / (PERIODS - 1)
    mH, mT = -0.3 - m * k(tH), -0.3 - m * k(tT)
    z = X[:, :2]
    lr = np.exp(-((z - mT) ** 2).sum(1) / 2 + ((z - mH) ** 2).sum(1) / 2)
    return np.where(g == 1, lr, 1.0)


def part3(n, reps, strengths=(0.4, 0.8, 1.6), sizes=(2000, 5000, 20000)):
    rows = []
    for spec in ("well-specified", "misspecified"):
        for m in strengths:
            for r in range(reps):
                rng = np.random.default_rng(9000 + r + int(100 * m) * 37 + (spec == "misspecified") * 7)
                data = [gen_cov(t, n, rng, m, spec) for t in range(PERIODS)]
                for t in range(2, PERIODS):
                    Xtr = np.vstack([data[s][0] for s in range(t - 1)])
                    ytr = np.concatenate([data[s][2] for s in range(t - 1)])
                    mod = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                    XT, gT, yT = data[t]
                    sT = mod.predict_proba(XT)[:, 1]
                    RG = year_stats(sT, gT, yT, 0.75)["RG"]
                    XH0, gH0, yH0 = data[t - 1]
                    for nH in sizes:
                        idx = np.random.default_rng(r * 31 + t).permutation(len(gH0))[:nH]
                        XH, gH, yH = XH0[idx], gH0[idx], yH0[idx]
                        sH = mod.predict_proba(XH)[:, 1]
                        e = estimates(XH, sH, gH, yH, XT, sT, gT, 0.75, seed=r, clf="lr",
                                      w_oracle=oracle_cov(XH, gH, t - 1, t, m))
                        rows.append(dict(spec=spec, strength=m, n_H=nH, rep=r, t=t, RG=RG,
                                         **{k: e[k] for k in EST}, ess=e["ess_frac"],
                                         ess_g1y1=e["ess_g1y1"]))
    return pd.DataFrame(rows)


def mae(df, by):
    out = df.groupby(by)[EST].apply(lambda d: (d.sub(df.loc[d.index, "RG"], axis=0)).abs().mean() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="synthetic_ablations")
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    f = open(f"{a.out}/summary.md", "w")
    f.write(f"# Synthetic ablations (reps = {a.reps}, n = {a.n})\n\n")
    if 1 in a.parts:
        p1 = part1(a.n, a.reps); p1.to_csv(f"{a.out}/part1.csv", index=False)
        bias = p1.groupby("scenario")[EST].apply(lambda d: (d.sub(p1.loc[d.index, "RG"], axis=0)).mean() * 100)
        f.write("## Part 1: mean absolute error (pp)\n\n" + mae(p1, "scenario").round(3).to_markdown() + "\n\n")
        f.write("## Part 1: mean signed bias (pp)\n\n" + bias.round(3).to_markdown() + "\n\n")
        f.write("## Part 1: ESS\n\n" + p1.groupby("scenario")[["ess", "ess_g1y1", "ess_noclip"]]
                .agg(["mean", "min"]).round(3).to_markdown() + "\n\n")
        print(f"part 1 done [{time.time() - t0:.0f}s]", flush=True)
    if 2 in a.parts:
        p2 = part2(5 * a.n, a.reps); p2.to_csv(f"{a.out}/part2.csv", index=False)
        tr = p2.groupby(["rule", "t"])[["G", "RG", "RC", "approval"]].mean().reset_index()
        tr[["G", "RG", "RC"]] *= 100
        sd = p2.groupby("rule").apply(lambda d: d.groupby("rep").RG.std().mean() * 100)
        f.write("## Part 2: label shift, fixed vs retrained rule (n = 5 x n per period; means over reps; pp)\n\n"
                + tr[tr.t.isin([2, PERIODS - 1])].round(2).to_markdown(index=False) + "\n\n"
                + "within-replicate SD of RG over periods (pp):\n\n" + sd.round(3).to_markdown() + "\n\n"
                + tr.round(2).to_markdown(index=False) + "\n\n")
        print(f"part 2 done [{time.time() - t0:.0f}s]", flush=True)
    if 3 in a.parts:
        p3 = part3(a.n, a.reps); p3.to_csv(f"{a.out}/part3.csv", index=False)
        f.write("## Part 3: MAE (pp) by specification, drift strength and reference size\n\n"
                + mae(p3, ["spec", "strength", "n_H"]).round(3).to_markdown() + "\n\n")
        f.write("## Part 3: ESS\n\n" + p3.groupby(["spec", "strength", "n_H"])[["ess", "ess_g1y1"]]
                .mean().round(3).to_markdown() + "\n\n")
        print(f"part 3 done [{time.time() - t0:.0f}s]", flush=True)
    f.write(f"Runtime {time.time() - t0:.0f}s\n")
    f.close()
    print(open(f"{a.out}/summary.md").read())


if __name__ == "__main__":
    main()
