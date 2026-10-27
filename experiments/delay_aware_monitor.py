#!/usr/bin/env python3
"""
Delay-aware residual monitoring (method component).

At decision time T the labels of year T are not yet observed; the most recent labelled
year is H = max{t <= T - d}. The approval gap G_T is computable from scores alone, but the
residual gap among qualified individuals,
    RG_T = P(approve | g=1, y=1) - P(approve | g=0, y=1),
needs labels. This script estimates RG_T at decision time with four estimators and
compares them with the realised RG_T once labels arrive.

  E0 stale          : RG of year H under year H's own thresholds (last known value)
  E1 soft-label     : a_g1 ~ sum(approve * s) / sum(s) on year T, raw scores as P(y=1|x)
  E2 recal. soft    : as E1 after group-wise isotonic recalibration of scores on year H
  E3 covariate-wt.  : current thresholds applied to year-H scores; group-wise approval rate
                      among qualified in H, importance-weighted by w(x,g) = p_T(x,g)/p_H(x,g)
                      from a cross-fitted domain classifier (the proposed estimator)

Year H is held out from training (model trained on years < H), so E2 and E3 use data the
model has not seen. The script also compares four approval-gap policies:
none, static (beta* of the first test year, frozen), carry-over (beta*_{T-1}),
per-period (beta*_T, computable at decision time).

Usage (needs run_temporal_audit.py in the same folder):
    python delay_aware_monitor.py --states CA --group race --task income --label-mode quantile
    python delay_aware_monitor.py --synthetic
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from run_temporal_audit import (TASKS, GROUP_COLS, load_real, make_synthetic, add_group,
                                build_model, coarsen, thresholds, approve, year_stats,
                                beta_star)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------------
def rule_thresholds(s, g, q, beta):
    """Group thresholds tau_g of the rule (beta-interpolated) on the scores s."""
    T, Tg = thresholds(s, g, q)
    return {k: (1 - beta) * T + beta * Tg[k] for k in (0, 1)}


def apply_tau(s, g, tau):
    return s >= np.where(g == 1, tau[1], tau[0])


def residual(a, g, w_y1):
    """Weighted approval-rate gap among qualified; w_y1 = weight on 'qualified' (0..1)."""
    num1, den1 = (a * w_y1)[g == 1].sum(), w_y1[g == 1].sum()
    num0, den0 = (a * w_y1)[g == 0].sum(), w_y1[g == 0].sum()
    return num1 / den1 - num0 / den0


def _domain_clf(kind, seed):
    if kind == "hgb":     # flexible; for categorical-coded tabular data such as ACS
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=seed)
    from sklearn.linear_model import LogisticRegression   # "lr": linear log density ratio
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def domain_weights(XH, gH, XT, gT, seed=0, clip_q=0.99, clf="hgb"):
    """Cross-fitted density ratio p_T(x,g) / p_H(x,g) evaluated on the rows of H.
    clf="lr" uses features [x, g, x*g] so the log ratio may differ by group."""
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
    w = p / (1 - p) * (len(ZH) / len(ZT))   # prior correction for unequal sizes
    w = np.minimum(w, np.quantile(w, clip_q))
    return w / w.mean()


def isotonic_by_group(sH, gH, yH, sT, gT):
    from sklearn.isotonic import IsotonicRegression
    out = np.empty(len(sT))
    for k in (0, 1):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(sH[gH == k], yH[gH == k])
        out[gT == k] = iso.predict(sT[gT == k])
    return out


def estimate_all(XH, sH, gH, yH, XT, sT, gT, q, beta, betaH, seed=0, clf="hgb", w_oracle=None):
    """All four estimates of RG_T, plus effective sample size of the E3 weights.
    w_oracle (synthetic only): true density ratio on H rows -> extra estimate E3_oracle."""
    tauT = rule_thresholds(sT, gT, q, beta)
    aT = apply_tau(sT, gT, tauT).astype(float)
    # E0: year H under its own rule
    aH_own = apply_tau(sH, gH, rule_thresholds(sH, gH, q, betaH)).astype(float)
    e0 = residual(aH_own, gH, (yH == 1).astype(float))
    # E1: raw scores as soft labels on year T
    e1 = residual(aT, gT, sT)
    # E2: group-wise isotonic recalibration on H, then soft labels
    e2 = residual(aT, gT, isotonic_by_group(sH, gH, yH, sT, gT))
    # E3: current rule applied to H, importance weighted towards year T
    w = domain_weights(XH, gH, XT, gT, seed, clf=clf)
    aH_cur = apply_tau(sH, gH, tauT).astype(float)
    e3 = residual(aH_cur, gH, w * (yH == 1))
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    out = dict(E0=e0, E1=e1, E2=e2, E3=e3, ess_frac=ess)
    if w_oracle is not None:
        out["E3_oracle"] = residual(aH_cur, gH, w_oracle * (yH == 1))
    return out


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
    ap.add_argument("--label-delay", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=["xgb"])
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.05, help="approval-gap tolerance")
    ap.add_argument("--delta-r", type=float, default=0.02, help="residual alert threshold")
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--data-dir", default="acs_data")
    ap.add_argument("--label-mode", default="nominal", choices=["nominal", "real", "quantile"])
    ap.add_argument("--base-year", type=int, default=2015)
    ap.add_argument("--label-quantile", type=float, default=0.6)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain-clf", default="hgb", choices=["hgb", "lr"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    tag = "synthetic" if a.synthetic else f"{a.task}_{a.group}_{'-'.join(a.states)}_d{a.label_delay}"
    if not a.synthetic and a.task == "income" and a.label_mode != "nominal":
        tag += f"_{a.label_mode}"
    if a.domain_clf == "lr":
        tag += "_lr"
    out = a.out or f"monitor_{tag}"
    os.makedirs(out, exist_ok=True)

    data = make_synthetic(a.years) if a.synthetic else load_real(
        a.task, a.states, a.years, a.data_dir, a.label_mode, a.base_year, a.label_quantile)
    data = {y: add_group(d, a.group) for y, d in data.items()}
    years = sorted(data)
    spec = TASKS[a.task]
    feats = [f for f in spec["features"] if f != GROUP_COLS[a.group]]

    rows = []
    for m in a.models:
        prev_beta, static_beta = None, None
        for T in years:
            labelled = [t for t in years if t <= T - a.label_delay]
            if len(labelled) < 2:
                continue                      # need a training year and a holdout year
            H = max(labelled)
            tr = pd.concat([data[t] for t in years if t < H], ignore_index=True)
            if len(tr) > a.max_train:
                tr = tr.sample(a.max_train, random_state=a.seed)
            model = build_model(m, feats, spec["categorical"], a.seed)
            model.fit(coarsen(tr[feats]), tr["__y"].values)
            dH, dT = data[H], data[T]
            XH, XT = coarsen(dH[feats]), coarsen(dT[feats])
            sH, sT = model.predict_proba(XH)[:, 1], model.predict_proba(XT)[:, 1]
            gH, yH = dH["__g"].values, dH["__y"].values
            gT, yT = dT["__g"].values, dT["__y"].values

            b_star = beta_star(sT, gT, a.q, a.delta)
            b_star = 0.0 if np.isnan(b_star) else b_star
            bH = beta_star(sH, gH, a.q, a.delta)
            bH = 0.0 if np.isnan(bH) else bH
            if static_beta is None:
                static_beta = b_star
            policies = {"none": 0.0, "static": static_beta,
                        "carry": 0.0 if prev_beta is None else prev_beta, "per_period": b_star}
            prev_beta = b_star

            # residual estimation under two rules: uncalibrated, and per-period parity calibration
            for rule, beta, betaH in (("beta0", 0.0, 0.0), ("beta_star", b_star, bH)):
                st = year_stats(sT, gT, yT, a.q, beta)
                est = estimate_all(XH.values, sH, gH, yH, XT.values, sT, gT, a.q, beta, betaH,
                                   a.seed, clf=a.domain_clf)
                rows.append(dict(model=m, year=T, holdout=H, kind="residual", rule=rule,
                                 beta=beta, true_RG=st["RG"], true_G=st["G"], **est))
            for pol, beta in policies.items():
                st = year_stats(sT, gT, yT, a.q, beta)
                frr = 1 - approve(sT, gT, a.q, beta)[yT == 1].mean()
                rows.append(dict(model=m, year=T, holdout=H, kind="policy", rule=pol, beta=beta,
                                 true_RG=st["RG"], true_G=st["G"], FRR=frr))
            print(f"  {m} T={T} H={H}: G={year_stats(sT, gT, yT, a.q)['G']:+.3f} "
                  f"RG={rows[-6]['true_RG']:+.3f} E0={rows[-6]['E0']:+.3f} "
                  f"E3={rows[-6]['E3']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/monitor_yearly.csv", index=False)

    # ---- summaries ---------------------------------------------------------------
    res = df[df.kind == "residual"].copy()
    err_rows, alert_rows = [], []
    for (m, rule), d in res.groupby(["model", "rule"]):
        truth = d.true_RG.values
        for e in ["E0", "E1", "E2", "E3"]:
            err = 100 * np.abs(d[e].values - truth)
            br = d.year.values == 2021
            t_al = np.abs(truth) > a.delta_r
            e_al = np.abs(d[e].values) > a.delta_r
            tp = (t_al & e_al).sum()
            err_rows.append(dict(model=m, rule=rule, estimator=e, MAE_pp=err.mean(),
                                 max_err_pp=err.max(),
                                 err_2021_pp=err[br].item() if br.any() else np.nan))
            alert_rows.append(dict(model=m, rule=rule, estimator=e, true_alerts=int(t_al.sum()),
                                   raised=int(e_al.sum()),
                                   precision=tp / e_al.sum() if e_al.sum() else np.nan,
                                   recall=tp / t_al.sum() if t_al.sum() else np.nan))
    errs, alerts = pd.DataFrame(err_rows), pd.DataFrame(alert_rows)
    pol = df[df.kind == "policy"]
    pol_sum = pol.groupby(["model", "rule"]).agg(
        years_outside=("true_G", lambda v: int((np.abs(v) > a.delta).sum())),
        max_abs_gap_pp=("true_G", lambda v: 100 * np.abs(v).max()),
        mean_FRR=("FRR", "mean")).reset_index()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = res[(res.model == a.models[0]) & (res.rule == "beta0")].sort_values("year")
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(d.year, 100 * d.true_RG, "k-o", lw=2, label="realised RG")
        for e, st in zip(["E0", "E1", "E2", "E3"], [":", "--", "-.", "-"]):
            ax.plot(d.year, 100 * d[e], st, marker=".", label=e)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(d.year.values); ax.set_xlabel("Test year")
        ax.set_ylabel("Residual gap (pp)"); ax.legend(ncol=3, fontsize=8)
        ax.set_title(f"{tag}: residual gap at decision time", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{out}/monitor_rg.png", dpi=150)
    except Exception as e:
        print("  figure skipped:", e)

    with open(f"{out}/monitor_summary.md", "w") as f:
        f.write(f"# Delay-aware residual monitoring: {tag}\n\n")
        f.write(f"- q = {a.q}; delta = {a.delta}; residual alert threshold = {a.delta_r}; "
                f"label delay = {a.label_delay}; domain classifier = {a.domain_clf}\n\n")
        f.write("- holdout year H = latest labelled year, excluded from training\n\n")
        f.write("## Estimation error (Table IV)\n\n" + errs.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Alert agreement (Table V)\n\n" + alerts.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Approval-gap policies (Table VI)\n\n" + pol_sum.round(4).to_markdown(index=False) + "\n\n")
        yr = res[res.rule == "beta0"][["model", "year", "holdout", "true_RG", "E0", "E1", "E2", "E3",
                                       "ess_frac"]]
        f.write("## Yearly residual estimates (beta = 0)\n\n" + yr.round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Runtime {time.time() - t0:.0f}s\n")
    print(open(f"{out}/monitor_summary.md").read())


if __name__ == "__main__":
    main()
