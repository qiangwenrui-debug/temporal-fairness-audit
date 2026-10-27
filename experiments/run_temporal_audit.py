#!/usr/bin/env python3
"""
Temporal fairness audit with yearly out-of-time retraining on Folktables (ACS PUMS).

Replicates the design of Paper 1 (Temporal Boundaries of Threshold Calibration) on a
public benchmark:
  * for each test year T, train on all years <= T - label_delay (expanding window)
  * score year-T individuals, approve the top-q share (score = P(favourable label))
  * yearly approval gap G_t = P_1t - P_0t (protected minus reference group)
  * Decomposition 1 (Eq. 2): pooled gap = sum_t w_1t G_t + sum_t (w_1t - w_0t) P_0t
  * Decomposition 2 (Eq. 3): G_t = RG_t + RC_t, RG_t = approval gap among qualified (y=1),
    |RC_t| <= max(1 - pi_1t, 1 - pi_0t)
  * beta*_t = minimum calibration strength with |G_t(beta)| <= delta,
    tau_g(beta) = (1 - beta) T_t(q) + beta T_gt(q)
  * carry-over test: apply beta*_{t-1} in year t
  * within-year bootstrap CIs (models held fixed, as in the paper)

Usage (real data; downloads ACS PUMS from census.gov on first run, then caches):
    pip install folktables xgboost lightgbm scikit-learn pandas matplotlib
    python run_temporal_audit.py --states CA --group race --task income
Smoke test without internet:
    python run_temporal_audit.py --synthetic
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# Task definitions (mirror folktables.ACSIncome / ACSEmployment)
# ----------------------------------------------------------------------------------
TASKS = {
    "income": {
        "features": ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"],
        "categorical": ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"],
        "favourable": "income > $50,000",
    },
    "employment": {
        "features": ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC",
                     "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"],
        "categorical": ["MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
                        "DEAR", "DEYE", "DREM", "SEX", "RAC1P"],
        "favourable": "employed",
    },
}
GROUP_COLS = {"sex": "SEX", "race": "RAC1P"}
# ACS 2019+ replaced RELP (0-17) with RELSHIPP (20-38); map back to the pre-2019 codes
RELSHIPP_TO_RELP = {20: 0, 21: 1, 23: 1, 22: 13, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
                    30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14, 36: 15, 37: 16, 38: 17}


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
# CPI-U, U.S. city average, annual averages (BLS series CUUR0000SA0, 1982-84 = 100).
# Verify against BLS before publication.
CPI_U = {2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
         2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702}


def income_label(filt, year, mode, base_year, quantile):
    """Income label under three definitions:
    nominal : PINCP > 50,000 (folktables default; threshold not inflation-adjusted)
    real    : PINCP in survey-year dollars (x ADJINC), deflated to base_year dollars, > 50,000
    quantile: real income above the year's own `quantile` (removes level drift entirely)
    """
    inc = filt["PINCP"].astype(float)
    if mode == "nominal":
        return (inc > 50000).astype(int)
    real = inc * filt["ADJINC"].astype(float) / 1e6 * CPI_U[base_year] / CPI_U[year]
    if mode == "real":
        return (real > 50000).astype(int)
    return (real > np.quantile(real, quantile)).astype(int)


def load_real(task, states, years, data_dir, label_mode="nominal", base_year=2015, label_q=0.6):
    import folktables
    problem = {"income": folktables.ACSIncome, "employment": folktables.ACSEmployment}[task]
    out = {}
    for y in years:
        try:
            src = folktables.ACSDataSource(survey_year=str(y), horizon="1-Year",
                                           survey="person", root_dir=data_dir)
            raw = src.get_data(states=states, download=True)
            if "RELP" not in raw.columns and "RELSHIPP" in raw.columns:
                raw["RELP"] = raw["RELSHIPP"].map(RELSHIPP_TO_RELP)   # 2019+ rename
            # apply the task's standard filter, then keep features + label
            filt = problem._preprocess(raw) if problem._preprocess else raw
            X = filt[problem.features].copy().fillna(-1)   # folktables' own postprocess
            if task == "income":
                yv = income_label(filt, y, label_mode, base_year, label_q)
            else:
                yv = problem.target_transform(filt[problem.target]) if problem.target_transform \
                    else filt[problem.target]
            X["__y"] = np.asarray(yv).astype(int)
            out[y] = X.reset_index(drop=True)
            print(f"  loaded {y}: {len(X):,} rows", flush=True)
        except Exception as e:  # e.g. 2020 has no standard 1-year file
            print(f"  skipped {y}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return out


def make_synthetic(years, n=20000, seed=0):
    """ACSIncome-shaped data with a planted, time-varying group relationship."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, y in enumerate(years):
        race = rng.choice([1, 2, 6, 8], size=n, p=[0.6, 0.08, 0.15, 0.17])
        sex = rng.choice([1, 2], size=n)
        age = rng.integers(17, 80, size=n)
        schl = rng.integers(1, 25, size=n)
        wkhp = rng.integers(1, 80, size=n)
        occp = rng.choice(np.arange(10, 9800, 10), size=n)
        cow = rng.integers(1, 9, size=n)
        mar = rng.integers(1, 6, size=n)
        relp = rng.integers(0, 18, size=n)
        pobp = rng.choice([6, 36, 48, 210, 303], size=n)
        black = (race == 2)
        # planted: group-correlated feature (cow==3) whose payoff reverses over time
        cow = np.where(black & (rng.random(n) < 0.5), 3, cow)
        trend = 1.5 - 3.0 * k / max(len(years) - 1, 1)  # +1.5 -> -1.5
        logit = (-6 + 0.25 * schl + 0.04 * wkhp + 0.03 * np.minimum(age, 55)
                 - 0.4 * (sex == 2) + trend * (cow == 3) + 0.1 * k)
        p = 1 / (1 + np.exp(-logit))
        lab = (rng.random(n) < p).astype(int)
        out[y] = pd.DataFrame({"AGEP": age, "COW": cow, "SCHL": schl, "MAR": mar, "OCCP": occp,
                               "POBP": pobp, "RELP": relp, "WKHP": wkhp, "SEX": sex,
                               "RAC1P": race, "__y": lab})
    return out


def add_group(df, group):
    if group == "sex":
        g = (df["SEX"] == 2).astype(int)              # female = 1, male = 0
        keep = np.ones(len(df), bool)
    else:
        g = (df["RAC1P"] == 2).astype(int)            # Black = 1, White = 0
        keep = df["RAC1P"].isin([1, 2]).values
    d = df.loc[keep].copy()
    d["__g"] = g[keep].values
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------
def build_model(name, feats, cats, seed):
    if name == "lr":
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        cat = [c for c in feats if c in cats]
        num = [c for c in feats if c not in cats]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
            ("num", StandardScaler(), num)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed,
                             n_jobs=-1, tree_method="hist", eval_metric="logloss")
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(name)


def coarsen(X):
    X = X.copy()
    if "OCCP" in X:
        X["OCCP"] = (X["OCCP"] // 100).astype(int)    # occupation major group
    if "POBP" in X:
        X["POBP"] = (X["POBP"] // 100).astype(int)    # place-of-birth region
    return X


# ----------------------------------------------------------------------------------
# Screening metrics
# ----------------------------------------------------------------------------------
def thresholds(s, g, q):
    """Approve the top-q share: threshold is the (1-q) quantile of scores."""
    T = np.quantile(s, 1 - q)
    Tg = {k: np.quantile(s[g == k], 1 - q) for k in (0, 1)}
    return T, Tg


def approve(s, g, q, beta, T=None, Tg=None):
    if T is None:
        T, Tg = thresholds(s, g, q)
    tau = np.where(g == 1, (1 - beta) * T + beta * Tg[1], (1 - beta) * T + beta * Tg[0])
    return s >= tau


def year_stats(s, g, y, q, beta=0.0):
    a = approve(s, g, q, beta)
    P1, P0 = a[g == 1].mean(), a[g == 0].mean()
    pi1, pi0 = y[g == 1].mean(), y[g == 0].mean()
    a11, a10 = a[(g == 1) & (y == 1)].mean(), a[(g == 1) & (y == 0)].mean()
    a01, a00 = a[(g == 0) & (y == 1)].mean(), a[(g == 0) & (y == 0)].mean()
    G = P1 - P0
    RG = a11 - a01                                             # gap among qualified
    RC = (1 - pi1) * (a10 - a11) - (1 - pi0) * (a00 - a01)     # composition component
    return dict(P1=P1, P0=P0, G=G, RG=RG, RC=RC, pi1=pi1, pi0=pi0,
                RC_bound=max(1 - pi1, 1 - pi0),
                FRR1=1 - a11, FRR0=1 - a01, FAR1=a10, FAR0=a00,
                identity_err=abs(G - RG - RC))


def gap_curve(s, g, q, betas):
    """Approval gap for every beta on a grid, vectorised with sorted scores."""
    T, Tg = thresholds(s, g, q)
    s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
    tau1 = (1 - betas) * T + betas * Tg[1]
    tau0 = (1 - betas) * T + betas * Tg[0]
    P1 = 1 - np.searchsorted(s1, tau1, side="left") / len(s1)
    P0 = 1 - np.searchsorted(s0, tau0, side="left") / len(s0)
    return P1 - P0


BETAS = np.linspace(0, 1, 1001)


def beta_star(s, g, q, delta):
    G = gap_curve(s, g, q, BETAS)
    ok = np.where(np.abs(G) <= delta)[0]
    return float(BETAS[ok[0]]) if len(ok) else np.nan


def boot_ci(s, g, y, q, B, rng):
    n = len(s)
    Gs, RGs = np.empty(B), np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        st = year_stats(s[i], g[i], y[i], q)
        Gs[b], RGs[b] = st["G"], st["RG"]
    return np.percentile(Gs, [2.5, 97.5]), np.percentile(RGs, [2.5, 97.5])


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="income", choices=list(TASKS))
    ap.add_argument("--group", default="race", choices=list(GROUP_COLS))
    ap.add_argument("--states", nargs="+", default=["CA"])
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    ap.add_argument("--label-delay", type=int, default=1,
                    help="train on years <= T - delay (1 = previous year; 2 mirrors the CSMAR design)")
    ap.add_argument("--models", nargs="+", default=["lr", "xgb", "lgbm"])
    ap.add_argument("--qs", nargs="+", type=float, default=[0.6, 0.75, 0.9])
    ap.add_argument("--main-q", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--use-group-feature", action="store_true",
                    help="keep the group attribute as a feature (default: drop it, as in the paper)")
    ap.add_argument("--data-dir", default="acs_data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-mode", default="nominal", choices=["nominal", "real", "quantile"],
                    help="income task only: nominal $50k, inflation-adjusted $50k, or within-year quantile")
    ap.add_argument("--base-year", type=int, default=2015)
    ap.add_argument("--label-quantile", type=float, default=0.6)
    a = ap.parse_args()

    t0 = time.time()
    tag = "synthetic" if a.synthetic else f"{a.task}_{a.group}_{'-'.join(a.states)}_d{a.label_delay}"
    if not a.synthetic and a.task == "income" and a.label_mode != "nominal":
        tag += f"_{a.label_mode}"
    out = a.out or f"results_{tag}"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"[1/4] Loading data ({tag})", flush=True)
    data = make_synthetic(a.years) if a.synthetic else load_real(
        a.task, a.states, a.years, a.data_dir, a.label_mode, a.base_year, a.label_quantile)
    data = {y: add_group(d, a.group) for y, d in data.items()}
    years = sorted(data)
    spec = TASKS[a.task]
    gcol = GROUP_COLS[a.group]
    feats = [f for f in spec["features"] if a.use_group_feature or f != gcol]

    test_years = [T for T in years if any(t <= T - a.label_delay for t in years)]
    print(f"  years available: {years}; test years: {test_years}", flush=True)
    if not test_years:
        raise SystemExit("No test years: need at least two loaded years (check 'skipped' messages above).")

    rows, scores = [], {}
    print("[2/4] Rolling retraining and scoring", flush=True)
    for m in a.models:
        for T in test_years:
            tr = pd.concat([data[t] for t in years if t <= T - a.label_delay], ignore_index=True)
            if len(tr) > a.max_train:
                tr = tr.sample(a.max_train, random_state=a.seed)
            te = data[T]
            Xtr, Xte = coarsen(tr[feats]), coarsen(te[feats])
            model = build_model(m, feats, spec["categorical"], a.seed)
            model.fit(Xtr, tr["__y"].values)
            s = model.predict_proba(Xte)[:, 1]
            g, y = te["__g"].values, te["__y"].values
            scores[(m, T)] = (s, g, y)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, s)
            # relative base rate in the training window vs test year (inherited relative risk)
            pi_tr = tr.groupby("__g")["__y"].mean()
            for q in a.qs:
                st = year_stats(s, g, y, q)
                bs = beta_star(s, g, q, a.delta)
                ciG, ciRG = boot_ci(s, g, y, q, a.boot, rng) if q == a.main_q else ((np.nan,) * 2,) * 2
                rows.append(dict(model=m, year=T, q=q, n=len(s), n1=int(g.sum()), auc=auc,
                                 beta_star=bs, G_lo=ciG[0], G_hi=ciG[1], RG_lo=ciRG[0], RG_hi=ciRG[1],
                                 pi_diff_train=pi_tr.get(1, np.nan) - pi_tr.get(0, np.nan),
                                 **st))
            print(f"  {m} {T}: AUC={auc:.3f}  G(q={a.main_q})="
                  f"{year_stats(s, g, y, a.main_q)['G']:+.3f}", flush=True)
    yearly = pd.DataFrame(rows)
    yearly["pi_diff_test"] = yearly["pi1"] - yearly["pi0"]
    yearly.to_csv(f"{out}/yearly.csv", index=False)

    print("[3/4] Pooled decomposition, carry-over, beta* sensitivity", flush=True)
    pooled_rows, carry_rows, sens_rows = [], [], []
    for m in a.models:
        for q in a.qs:
            yq = yearly[(yearly.model == m) & (yearly.q == q)].sort_values("year")
            n1 = yq["n1"].values.astype(float)
            n0 = (yq["n"] - yq["n1"]).values.astype(float)
            w1, w0 = n1 / n1.sum(), n0 / n0.sum()
            P1p, P0p = (w1 * yq.P1).sum(), (w0 * yq.P0).sum()
            Gp = P1p - P0p
            wmean = (w1 * yq.G).sum()
            comp = ((w1 - w0) * yq.P0).sum()
            # pooled CI: resample individuals within every year
            Bs = []
            if q == a.main_q:
                for b in range(a.boot):
                    P1b = P0b = N1 = N0 = 0.0
                    for T in yq.year:
                        s, g, y = scores[(m, T)]
                        i = rng.integers(0, len(s), len(s))
                        ap_ = approve(s[i], g[i], q, 0.0)
                        gi = g[i]
                        P1b += ap_[gi == 1].sum(); N1 += (gi == 1).sum()
                        P0b += ap_[gi == 0].sum(); N0 += (gi == 0).sum()
                    Bs.append(P1b / N1 - P0b / N0)
            lo, hi = (np.percentile(Bs, [2.5, 97.5]) if Bs else (np.nan, np.nan))
            absG = yq.G.abs()
            pooled_rows.append(dict(
                model=m, q=q, pooled_gap=Gp, pooled_lo=lo, pooled_hi=hi,
                weighted_mean_yearly=wmean, composition=comp, identity_err=abs(Gp - wmean - comp),
                mean_abs_yearly=absG.mean(), sd_yearly=yq.G.std(ddof=1),
                min_yearly=yq.G.min(), max_yearly=yq.G.max(),
                yrs_out_3=int((absG > .03).sum()), yrs_out_5=int((absG > .05).sum()),
                yrs_out_10=int((absG > .10).sum()), n_years=len(yq),
                yrs_sig=int(((yq.G_lo > 0) | (yq.G_hi < 0)).sum()) if q == a.main_q else np.nan,
                corr_gap_pidiff_test=np.corrcoef(yq.G, yq.pi_diff_test)[0, 1],
                corr_RG_pidiff_train=np.corrcoef(yq.RG, yq.pi_diff_train)[0, 1],
                corr_RG_pidiff_test=np.corrcoef(yq.RG, yq.pi_diff_test)[0, 1],
                beta_min=yq.beta_star.min(), beta_max=yq.beta_star.max(),
                max_abs_RC=yq.RC.abs().max(), max_identity_err_yearly=yq.identity_err.max()))
            # carry-over test
            prev = None
            for T in yq.year:
                s, g, y = scores[(m, T)]
                if prev is not None:
                    b_used = 0.0 if np.isnan(prev) else prev
                    a_ = approve(s, g, q, b_used)
                    Gc = a_[g == 1].mean() - a_[g == 0].mean()
                    carry_rows.append(dict(model=m, q=q, year=T, beta_used=b_used, gap=Gc,
                                           outside=abs(Gc) > a.delta))
                prev = yq.loc[yq.year == T, "beta_star"].item()
        # pooled beta* over a q-grid: same beta applied in every year, pooled gap
        for q in np.round(np.arange(0.60, 0.851, 0.05), 2):
            curves, w1s = [], []
            num1 = np.zeros_like(BETAS); num0 = np.zeros_like(BETAS); N1 = N0 = 0
            for T in test_years:
                s, g, y = scores[(m, T)]
                T_, Tg = thresholds(s, g, q)
                s1, s0 = np.sort(s[g == 1]), np.sort(s[g == 0])
                num1 += len(s1) - np.searchsorted(s1, (1 - BETAS) * T_ + BETAS * Tg[1])
                num0 += len(s0) - np.searchsorted(s0, (1 - BETAS) * T_ + BETAS * Tg[0])
                N1 += len(s1); N0 += len(s0)
            Gb = num1 / N1 - num0 / N0
            ok = np.where(np.abs(Gb) <= a.delta)[0]
            sens_rows.append(dict(model=m, q=q, pooled_gap=Gb[0],
                                  pooled_beta_star=float(BETAS[ok[0]]) if len(ok) else np.nan))
    pooled = pd.DataFrame(pooled_rows); pooled.to_csv(f"{out}/pooled.csv", index=False)
    carry = pd.DataFrame(carry_rows); carry.to_csv(f"{out}/carryover.csv", index=False)
    sens = pd.DataFrame(sens_rows); sens.to_csv(f"{out}/pooled_beta_by_q.csv", index=False)

    print("[4/4] Figures and summary", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        yq = yearly[yearly.q == a.main_q]
        for m, d in yq.groupby("model"):
            d = d.sort_values("year")
            ax[0].errorbar(d.year, 100 * d.G, yerr=[100 * (d.G - d.G_lo), 100 * (d.G_hi - d.G)],
                           marker="o", capsize=3, label=m)
            ax[1].errorbar(d.year, 100 * d.RG, yerr=[100 * (d.RG - d.RG_lo), 100 * (d.RG_hi - d.RG)],
                           marker="o", capsize=3, label=m)
        for x, t in zip(ax, ["Approval gap G_t", "Residual gap RG_t (among qualified)"]):
            x.axhspan(-100 * a.delta, 100 * a.delta, color="0.85", zorder=0)
            x.axhline(0, color="0.4", lw=0.8)
            x.set_title(t); x.set_xlabel("Test year")
            x.set_xticks(sorted(yq.year.unique()))
        ax[0].set_ylabel("Percentage points (group 1 − group 0)")
        ax[0].legend()
        fig.suptitle(f"{tag}, q = {a.main_q}; shaded = ±{100 * a.delta:.0f} pp band")
        fig.tight_layout(); fig.savefig(f"{out}/yearly_gaps.png", dpi=150)
    except Exception as e:
        print("  figure skipped:", e)

    with open(f"{out}/summary.md", "w") as f:
        f.write(f"# Temporal fairness audit: {tag}\n\n")
        f.write(f"- task: {spec['favourable']}; group: {a.group} "
                f"({'female vs male' if a.group == 'sex' else 'Black vs White'}); "
                f"states: {a.states}; label delay: {a.label_delay}\n")
        f.write(f"- test years: {test_years}; delta = {a.delta}; bootstrap B = {a.boot}\n")
        f.write(f"- group feature used: {a.use_group_feature}\n\n")
        cols = ["model", "q", "pooled_gap", "pooled_lo", "pooled_hi", "weighted_mean_yearly",
                "composition", "mean_abs_yearly", "sd_yearly", "min_yearly", "max_yearly",
                "yrs_out_3", "yrs_out_5", "yrs_out_10", "yrs_sig", "beta_min", "beta_max",
                "corr_gap_pidiff_test", "max_abs_RC"]
        f.write("## Pooled vs yearly\n\n" + pooled[cols].round(4).to_markdown(index=False) + "\n\n")
        yq = yearly[yearly.q == a.main_q][["model", "year", "auc", "pi1", "pi0", "G", "G_lo", "G_hi",
                                            "RG", "RC", "RC_bound", "beta_star"]]
        f.write(f"## Yearly (q = {a.main_q})\n\n" + yq.round(4).to_markdown(index=False) + "\n\n")
        cs = carry.groupby(["model", "q"]).agg(years_outside=("outside", "sum"),
                                                n=("outside", "size"),
                                                max_abs_gap=("gap", lambda v: v.abs().max())).reset_index()
        f.write("## Carry-over of previous year's beta*\n\n" + cs.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Pooled beta* by approval rate\n\n" + sens.round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Identity checks: max |G - RG - RC| = {yearly.identity_err.max():.2e}; "
                f"max |pooled - wmean - comp| = {pooled.identity_err.max():.2e}\n\n")
        f.write(f"Environment: Python {platform.python_version()}, {platform.system()} "
                f"{platform.machine()}; runtime {time.time() - t0:.0f}s\n")
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
