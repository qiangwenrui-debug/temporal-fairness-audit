#!/usr/bin/env python3
"""
Robustness checks for the California income x race case (quantile label, d = 1, q = 0.75):
  base     : main specification (unweighted, all features)
  weighted : ACS person weights PWGTP in thresholds, approval rates, residuals and the
             within-year income quantile; models trained with sample weights
  no_occp  : occupation (OCCP) removed from the features (2018 coding change)
Needs run_temporal_audit.py in the same folder; ACS files are read from acs_data/.
Usage:  python robustness_checks.py            (about 3-5 minutes)
        python robustness_checks.py --synthetic (quick self-test)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from run_temporal_audit import (TASKS, RELSHIPP_TO_RELP, CPI_U, build_model, coarsen,
                                make_synthetic)

warnings.filterwarnings("ignore")
BETAS = np.linspace(0, 1, 1001)


# ---------------------------------------------------------------- weighted primitives
def wquantile(x, w, p):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(p, c, x)


def rates(s, g, y, w, q, beta):
    T = wquantile(s, w, 1 - q)
    Tg = {k: wquantile(s[g == k], w[g == k], 1 - q) for k in (0, 1)}
    tau = np.where(g == 1, (1 - beta) * T + beta * Tg[1], (1 - beta) * T + beta * Tg[0])
    a = (s >= tau).astype(float)
    m = lambda mask: (a * w)[mask].sum() / w[mask].sum()
    G = m(g == 1) - m(g == 0)
    RG = m((g == 1) & (y == 1)) - m((g == 0) & (y == 1))
    return G, RG


def beta_star(s, g, y, w, q, delta):
    for b in BETAS[::5]:                               # coarse pass, step 0.005
        if abs(rates(s, g, y, w, q, b)[0]) <= delta:
            lo = max(b - 0.005, 0)
            for bb in BETAS[(BETAS >= lo) & (BETAS <= b)]:
                if abs(rates(s, g, y, w, q, bb)[0]) <= delta:
                    return bb
    return np.nan


# ---------------------------------------------------------------- data
def load(states, years, data_dir, synthetic=False):
    if synthetic:
        out = {}
        for y, d in make_synthetic(years).items():
            d = d.copy()
            d["PWGTP"] = np.random.default_rng(y).integers(5, 200, len(d))
            d["__inc"] = d["__y"] * 60000 + np.random.default_rng(y).normal(40000, 10000, len(d))
            out[y] = d
        return out
    import folktables
    prob = folktables.ACSIncome
    out = {}
    for y in years:
        src = folktables.ACSDataSource(survey_year=str(y), horizon="1-Year", survey="person",
                                       root_dir=data_dir)
        raw = src.get_data(states=states, download=True)
        if "RELP" not in raw.columns and "RELSHIPP" in raw.columns:
            raw["RELP"] = raw["RELSHIPP"].map(RELSHIPP_TO_RELP)
        f = prob._preprocess(raw)
        X = f[prob.features].copy().fillna(-1)
        X["PWGTP"] = f["PWGTP"].astype(float).values
        X["__inc"] = (f["PINCP"].astype(float) * f["ADJINC"].astype(float) / 1e6
                      * CPI_U[2015] / CPI_U[y]).values
        out[y] = X.reset_index(drop=True)
        print(f"  loaded {y}: {len(X):,}", flush=True)
    return out


def prepare(d, weighted, label_q=0.6):
    d = d[d["RAC1P"].isin([1, 2])].copy()
    d["__g"] = (d["RAC1P"] == 2).astype(int)
    w = d["PWGTP"].values if weighted else np.ones(len(d))
    thr = wquantile(d["__inc"].values, w, label_q) if weighted else np.quantile(d["__inc"], label_q)
    d["__y"] = (d["__inc"] > thr).astype(int)
    d["__w"] = w
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["CA"])
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    ap.add_argument("--models", nargs="+", default=["xgb"])
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--data-dir", default="acs_data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="robustness_income_race_CA")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = load(a.states, a.years, a.data_dir, a.synthetic)
    years = sorted(raw)
    base_feats = [f for f in TASKS["income"]["features"] if f != "RAC1P"]
    specs = {"base": (False, base_feats), "weighted": (True, base_feats),
             "no_occp": (False, [f for f in base_feats if f != "OCCP"])}

    rows = []
    for spec, (weighted, feats) in specs.items():
        data = {y: prepare(raw[y], weighted) for y in years}
        for m in a.models:
            for T in years[1:]:
                tr = pd.concat([data[t] for t in years if t <= T - 1], ignore_index=True)
                if len(tr) > a.max_train:
                    tr = tr.sample(a.max_train, random_state=0)
                model = build_model(m, feats, TASKS["income"]["categorical"], 0)
                if weighted:
                    key = "clf__sample_weight" if m == "lr" else "sample_weight"
                    model.fit(coarsen(tr[feats]), tr["__y"].values, **{key: tr["__w"].values})
                else:
                    model.fit(coarsen(tr[feats]), tr["__y"].values)
                te = data[T]
                s = model.predict_proba(coarsen(te[feats]))[:, 1]
                g, y, w = te["__g"].values, te["__y"].values, te["__w"].values
                G, RG = rates(s, g, y, w, a.q, 0.0)
                b = beta_star(s, g, y, w, a.q, a.delta)
                rows.append(dict(spec=spec, model=m, year=T, n=len(te), w1=w[g == 1].sum(),
                                 w0=w[g == 0].sum(), G=G, RG=RG, beta_star=b))
                print(f"  {spec:8s} {m} {T}: G={100*G:+.1f} RG={100*RG:+.1f} beta*={b:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/robustness_yearly.csv", index=False)

    # pooled gap with group weights over years (Prop. 1 weights) and summary
    summ = []
    for (spec, m), d in df.groupby(["spec", "model"], sort=False):
        w1, w0 = d.w1 / d.w1.sum(), d.w0 / d.w0.sum()
        pre, post = d[d.year <= 2019], d[d.year >= 2021]
        summ.append(dict(spec=spec, model=m,
                         mean_gap_2015_19_pp=100 * pre.G.mean(), mean_gap_2021_23_pp=100 * post.G.mean(),
                         pooled_gap_pp=100 * ((w1 * d.G).sum()),
                         RG_2015_18_pp=f"{100*d[d.year <= 2018].RG.min():.1f} to {100*d[d.year <= 2018].RG.max():.1f}",
                         RG_2021_23_pp=f"{100*post.RG.min():.1f} to {100*post.RG.max():.1f}",
                         years_outside_5=int((d.G.abs() > a.delta).sum()), n_years=len(d),
                         beta_star_range=f"{d.beta_star.min():.2f}-{d.beta_star.max():.2f}"))
    S = pd.DataFrame(summ)
    with open(f"{a.out}/robustness_summary.md", "w") as f:
        f.write("# Robustness: CA income x race, quantile label, d = 1, q = 0.75\n\n")
        f.write("pooled_gap_pp is the weighted average of yearly gaps (Prop. 1 without the "
                "composition term).\n\n")
        f.write(S.round(2).to_markdown(index=False) + "\n\n")
        piv = df.pivot_table(index="year", columns=["spec", "model"], values="G").mul(100).round(1)
        f.write("## Yearly approval gap (pp)\n\n" + piv.to_markdown() + "\n")
    print(open(f"{a.out}/robustness_summary.md").read())


if __name__ == "__main__":
    main()
