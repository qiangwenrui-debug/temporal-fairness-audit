"""Collects the census-suite and synthetic-ablation results into collected_tables.json (table cells) and collected_numbers.json
(numbers quoted in the text as «KEY»).

    python collect_results.py SUITE_DIR SYN_DIR SYN_P2_DIR [OTHER_MACHINE_AUDIT_YEARLY MAC_AUDIT_YEARLY]

SUITE_DIR   census_suite/ written by run_census_suite.py (census, XGBoost)
SYN_DIR     synthetic_ablations/ (parts 1 and 3), SYN_P2_DIR synthetic_ablations_p2/ (part 2)
The optional pair of yearly.csv files (same audit on two machines) gives the hardware note."""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

REV, SYN, SYN2 = sys.argv[1:4]
M = "−"
Q = 0.75


def r(x, nd):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def f(x, nd=2, plus=False):
    v = r(x, nd)
    s = f"{abs(v):.{nd}f}"
    return (M + s) if v < 0 else (("+" + s) if (plus and v > 0) else s)


def rng(a, b, nd=1):
    lo, hi = sorted([a, b], key=lambda v: v)
    # print as "−8.7 to −9.7" ordered from smaller to larger magnitude for negatives
    if lo < 0 and hi < 0:
        lo, hi = hi, lo
    return f"{f(lo, nd)} to {f(hi, nd)}"


def load(name, fn):
    return pd.read_csv(os.path.join(REV, name, fn))


N, TAB = {}, {}

# ------------------------------------------------------------------ monitoring, California
def yrs_txt(years):
    years = sorted(years)
    pre = [y for y in years if y <= 2019]; post = [y for y in years if y >= 2021]
    part = lambda v: (f"{v[0]}–{v[-1]}" if len(v) > 1 else f"{v[0]}") if v else ""
    return ", ".join(p for p in [part(pre), part(post)] if p)


def loadm(name):
    """Monitoring estimates with E0 = last observed residual (realized RG of the reference year
    under the rule deployed in that year) and E0p = current model with reference-year thresholds."""
    d = load(name, "estimates.csv").rename(columns={"E0": "E0p"})
    d["E0"] = d.holdout.map(dict(zip(d.year, d.true_RG)))
    return d


E = ["E0", "E0p", "E1", "E2", "E3u", "E3"]
ca = loadm("monitor_CA_d1")
hgb = loadm("monitor_CA_d1_hgb")
d2 = loadm("monitor_CA_d2")
emp = loadm("monitor_emp_race_CA_d1")
y21 = ca[ca.year == 2021].iloc[0]
N["CA_RG_2021"] = f(100 * y21.true_RG, 2)
for e, k in [("E0", "E0"), ("E0p", "E0P"), ("E1", "E1"), ("E2", "E2"), ("E3u", "E3U"), ("E3", "E3")]:
    N[f"CA_{k}_2021_PP"] = f(100 * y21[e], 2)
    N[f"CA_{k}_2021"] = f"{r(100 * y21[e] / y21.true_RG, 0):.0f}"
    N[f"CA_{k}_MAE"] = f(100 * (ca[e] - ca.true_RG).abs().mean(), 2)
cc = ca[ca.E0.notna()]
for e, k in [("E0", "E0"), ("E0p", "E0P"), ("E1", "E1"), ("E2", "E2"), ("E3u", "E3U"), ("E3", "E3")]:
    N[f"CA_{k}_MAE"] = f(100 * (cc[e] - cc.true_RG).abs().mean(), 2)
    N[f"CA_{k}_MAX"] = f(100 * (cc[e] - cc.true_RG).abs().max(), 2)
N["CA_N_COMMON"] = str(len(cc))
N["CA_COMMON_YEARS"] = yrs_txt(cc.year)
N["CA_RULE_SHARE"] = f"{r(100 * (y21.E3u - y21.E0p) / (y21.E3 - y21.E0p), 0):.0f}"
N["CA_N_E0_YEARS"] = str(int(ca.E0.notna().sum()))
N["CA_NG1Y1_MIN"] = f"{ca.n_H_g1y1.min():,}"
N["CA_NG1Y1_MAX"] = f"{ca.n_H_g1y1.max():,}"
N["CA_ESS_G1Y1_2021"] = f"{r(100 * y21.ess_g1y1, 0):.0f}"
N["CA_ESS_G1Y1_MIN"] = f"{r(100 * ca.ess_g1y1.min(), 0):.0f}"
N["CA_NOCLIP_MAXDIFF"] = f(100 * (ca.E3_noclip - ca.E3).abs().max(), 2)
N["CA_C95_MAXDIFF"] = f(100 * (ca.E3_c95 - ca.E3).abs().max(), 2)
h21 = hgb[hgb.year == 2021].iloc[0]
N["CA_HGB_E3_2021_PP"] = f(100 * h21.E3, 2)
N["CA_HGB_E3_2021"] = f"{r(100 * h21.E3 / h21.true_RG, 0):.0f}"
xy = sorted(set(ca.loc[ca.E0.notna(), "year"]) & set(d2.loc[d2.E0.notna(), "year"]))
N["X_YEARS"] = yrs_txt(xy)
for e, k in [("E0", "E0"), ("E0p", "E0P"), ("E2", "E2"), ("E3", "E3"), ("E3u", "E3U")]:
    for nm, dd in [("X_D1", ca), ("X_D2", d2)]:
        c = dd[dd.year.isin(xy)]
        N[f"{nm}_{k}"] = f(100 * (c[e] - c.true_RG).abs().mean(), 2)
for e, k in [("E0", "E0"), ("E0p", "E0P"), ("E2", "E2"), ("E3", "E3"), ("E3u", "E3U")]:
    N[f"D2_{k}_MAE"] = f(100 * (d2[e] - d2.true_RG).abs().mean(), 2)
y22 = d2[d2.year == 2022].iloc[0]
N["D2_E3_2022_PP"] = f(100 * y22.E3, 2)
N["D2_RG_2022"] = f(100 * y22.true_RG, 2)
N["D2_E0_2022_PP"] = f(100 * y22.E0, 2)
N["D2_E0P_2022_PP"] = f(100 * y22.E0p, 2)
N["D2_N_E0_YEARS"] = str(int(d2.E0.notna().sum()))


def err_rows(df, ests, boosted=None):
    """Metrics on the common years in which E0 exists; last column: MAE over all test years."""
    out = []
    lab = {"E0": "E0, last observed", "E0p": "E0′, current model", "E3": "E3 (logistic)"}
    common = set(df.loc[df.E0.notna(), "year"])
    for name, col, full in [(e, e, df) for e in ests] + ([("E3 (boosted)", "E3", boosted)] if boosted is not None else []):
        src = full[full.year.isin(common)]
        allv = full[full[col].notna()]
        t_al = src.true_RG.abs() > 0.02
        err = 100 * (src[col] - src.true_RG).abs()
        al = src[col].abs() > 0.02
        tp = int((al & (src.true_RG.abs() > 0.02)).sum())
        e21 = err[src.year == 2021]
        out.append([lab.get(name, name), f"{r(err.mean(), 2):.2f}",
                    f"{r(err.max(), 2):.2f}", f"{r(e21.iloc[0], 2):.2f}" if len(e21) else "–",
                    str(int(al.sum())), f"{tp / al.sum():.2f}" if al.sum() else "–",
                    f"{tp / t_al.sum():.2f}" if t_al.sum() else "–",
                    "–" if name == "E0" else f"{r(100 * (allv[col] - allv.true_RG).abs().mean(), 2):.2f}"])
    return out


TAB["fig4"] = {"years": ca.year.tolist(), "true": ca.true_RG.tolist(), **{e: ca[e].tolist() for e in E}}
TAB["monitor"] = []
for label, df, b in [("CA income × race, *d* = 1", ca, hgb), ("CA income × race, *d* = 2", d2, None),
                     ("CA employment × race, *d* = 1", emp, None)]:
    cm = df[df.E0.notna()]
    TAB["monitor"].append({"group": f"{label}: common years {yrs_txt(cm.year)} ({len(cm)}; years with |*RG*_{{t}}| > 2 pp: {int((cm.true_RG.abs() > 0.02).sum())})"})
    TAB["monitor"] += err_rows(df, E, b)

# other states (Table A5)
st_rows, agg = [], {k: [] for k in ["E0", "E0p", "E2", "E3u", "E3"]}
true_alerts = years = fa = miss = 0
for s in ["TX", "NY", "FL"]:
    d = loadm(f"monitor_{s}_d1")
    ess_min = d.ess_g1y1.min()
    d = d[d.E0.notna()]                                  # common years
    t_al = d.true_RG.abs() > 0.02
    row = [s, f"{int(t_al.sum())}/{len(d)}"]
    for e in ["E0", "E0p", "E1", "E2", "E3u", "E3"]:
        row.append(f"{r(100 * (d[e] - d.true_RG).abs().mean(), 2):.2f}")
        if e in agg:
            agg[e].append(100 * (d[e] - d.true_RG).abs())
    a3 = d.E3.abs() > 0.02
    row.append(f"{int(a3.sum())} / {int((a3 & t_al).sum())}")
    row.append(f"{r(100 * (d.E3 - d.true_RG).abs().max(), 2):.2f}")
    row.append(f"{r(100 * ess_min, 0):.0f}")
    st_rows.append(row)
    true_alerts += int(t_al.sum()); years += len(d)
    fa += int((a3 & ~t_al).sum()); miss += int((~a3 & t_al).sum())
TAB["states"] = st_rows
words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
N["ST_TRUE_ALERTS"], N["ST_YEARS"] = str(true_alerts), str(years)
best, beats, ess21 = [], 0, []
full = {"TX": "Texas", "NY": "New York", "FL": "Florida"}
nm = {"E0": "E0", "E0p": "E0′", "E1": "E1", "E2": "E2", "E3u": "E3u", "E3": "E3"}
for s_ in ["TX", "NY", "FL"]:
    d = loadm(f"monitor_{s_}_d1")
    ess21.append(d.loc[d.year == 2021, "ess_frac"].iloc[0])
    d = d[d.E0.notna()]
    maes = {e: (d[e] - d.true_RG).abs().mean() for e in nm}
    best.append(f"{nm[min(maes, key=maes.get)]} in {full[s_]}")
    beats += maes["E3"] < maes["E0"]
    agg.setdefault("E1", []).append(100 * (d.E1 - d.true_RG).abs())
N["ST_BEST"] = ", ".join(best[:-1]) + " and " + best[-1]
N["ST_E3_BEATS_E0"] = {0: "none", 1: "one", 2: "two", 3: "all three"}[beats]
N["ST_E1_MAE"] = f(pd.concat(agg["E1"]).mean(), 2)
N["ST_ESS_2021_MIN"] = f"{r(100 * min(ess21), 0):.0f}"
fa_t = {0: "raised no false alert", 1: "raised one false alert"}.get(fa, f"raised {words.get(fa, fa)} false alerts")
miss_t = {0: "missed none", 1: "missed one"}.get(miss, f"missed {words.get(miss, miss)}")
N["ST_E3_ALERT_TXT"] = f"{fa_t} and {miss_t}"
for e, k in [("E0", "E0"), ("E0p", "E0P"), ("E2", "E2"), ("E3u", "E3U"), ("E3", "E3")]:
    N[f"ST_{k}_MAE"] = f(pd.concat(agg[e]).mean(), 2)
N["ST_E3_FA"], N["ST_E3_MISS"] = words.get(fa, str(fa)), words.get(miss, str(miss))

# ------------------------------------------------------------------ policies (Table VIII)
names = {"none": "No calibration", "static": "Static", "carry": "Carry-over", "per_period": "Per-period",
         "per_period_budget": "Per-period, fixed budget", "pooled_retro": "Pooled (retrospective)"}
TAB["policy"], dev, dfrr, dfar, ties = [], [], [], [], []
for label, folder in [("CA income × race", "monitor_CA_d1"), ("CA employment × race", "monitor_emp_race_CA_d1")]:
    pol = load(folder, "policies.csv")
    TAB["policy"].append({"group": label})
    base = pol[pol.policy == "none"].set_index("year")
    for p, name in names.items():
        d = pol[pol.policy == p]
        out = int((d.G.abs() > 0.05 + 1e-12).sum())
        TAB["policy"].append([name, f"{out}/{len(d)}", f"{r(100 * d.G.abs().max(), 1):.1f}",
                              f"{r(100 * d.approval.min(), 2):.2f}–{r(100 * d.approval.max(), 2):.2f}",
                              f"{r(100 * d.FRR.mean(), 2):.2f}", f"{r(100 * d.FAR.mean(), 2):.2f}",
                              f"{r(100 * d.RG.abs().max(), 1):.1f}"])
        dev.append((d.approval - Q).abs().max())
        if p == "per_period":
            dd = d.set_index("year")
            dfrr.append((dd.FRR - base.FRR).abs().max()); dfar.append((dd.FAR - base.FAR).abs().max())
    ties.append(pol.tie_share.max())
N["POL_APPR_DEV"] = f(100 * max(dev), 2)
N["POL_FRR_DIFF"] = f(100 * max(dfrr), 2)
N["POL_FAR_DIFF"] = f(100 * max(dfar), 2)

# ------------------------------------------------------------------ audit protocol (Table A3)
labels = {"audit_inc_race_CA": "Income × race, CA", "audit_inc_race_TX-NY-FL": "Income × race, TX/NY/FL",
          "audit_emp_race_CA": "Employment × race, CA", "audit_inc_sex_CA": "Income × sex, CA",
          "audit_emp_sex_CA": "Employment × sex, CA"}
TAB["bpool"], comp, adev, bdiff = [], [], [], []
for folder, label in labels.items():
    y = load(folder, "yearly.csv"); p = load(folder, "pooled.csv").iloc[0]
    n = len(y)
    TAB["bpool"].append([label, f"{r(p.beta_pool, 2):.2f}", f"{int((y.G0.abs() <= 0.05 + 1e-12).sum())}/{n}",
                         f"{int((y.G_bpool.abs() <= 0.05 + 1e-12).sum())}/{n}",
                         f"{f(100 * y.G_bpool.min(), 1)} to {f(100 * y.G_bpool.max(), 1)}",
                         f"{r(100 * (y.approval_bstar - Q).abs().max(), 2):.2f}",
                         f"{r(100 * (y.approval_bpool - Q).abs().max(), 2):.2f}",
                         f"{r((y.beta_star - y.beta_star_budget).abs().max(), 3):.3f}",
                         f"{f(100 * p.composition, 2, plus=True)}"])
    comp.append(abs(p.composition)); adev += [(y.approval_bstar - Q).abs().max(), (y.approval_bpool - Q).abs().max()]
    bdiff.append((y.beta_star - y.beta_star_budget).abs().max())
    ties.append(y.tie_share.max())
    if folder == "audit_inc_race_CA":
        pre, post = y[y.year <= 2019], y[y.year >= 2021]
        N["CA_BPOOL_PRE"] = rng(100 * pre.G_bpool.max(), 100 * pre.G_bpool.min())
        N["CA_BPOOL_POST"] = rng(100 * post.G_bpool.max(), 100 * post.G_bpool.min())
    if folder == "audit_inc_sex_CA":
        N["SEX_BPOOL_OUT"] = str(int((y.G_bpool.abs() > 0.05 + 1e-12).sum()))
N["TIE_MAX"] = f"{r(100 * max(ties), 2):.2f}"
_bd = []
for _f in ["monitor_CA_d1", "monitor_CA_d2", "monitor_TX_d1", "monitor_NY_d1", "monitor_FL_d1", "monitor_emp_race_CA_d1", "monitor_inc_sex_CA_d1"]:
    _p = load(_f, "policies.csv"); _bd.append((_p[_p.policy.str.contains("budget")].approval - Q).abs().max())
for _f in labels:
    _bd.append((load(_f, "yearly.csv").approval_bstar_budget - Q).abs().max())
N["BUDGET_DEV"] = f"{r(100 * max(_bd), 2):.2f}"
N["COMP_MAX_ACS"] = f(100 * max(comp), 2)
N["APPR_DEV_MAX"] = f(100 * max(adev), 2)
N["BUDGET_BDIFF"] = f(max(bdiff), 3)

# ------------------------------------------------------------------ robustness (Table V, last rows)
rb = load("robustness_CA", "robustness_summary.csv").set_index("spec")
TAB["robust_extra"] = []
for spec, lab in [("base", "Top 40% within Black and White (check baseline)"),
                  ("weighted", "  with survey weights (PWGTP)"), ("no_occp", "  without occupation (OCCP)")]:
    s = rb.loc[spec]
    b0, b1 = [float(v) for v in str(s.beta_star_range).split("-")]
    TAB["robust_extra"].append([lab, "1", f(s.mean_gap_2015_19_pp, 1), f(s.mean_gap_2021_23_pp, 1),
                                f(s.pooled_gap_pp, 1), f"{b0:.2f}–{b1:.2f}", f"{s.beta_pool:.2f}"])

# ------------------------------------------------------------------ synthetic
EST = ["E0", "E1", "E2", "E3u", "E3", "E3_c95", "E3_c999", "E3_noclip", "E3_oracle"]
p1 = pd.read_csv(os.path.join(SYN, "part1.csv"))
cov = p1[p1.scenario == "covariate"]
N["S_E3U_COV"] = f(100 * (cov.E3u - cov.RG).mean(), 2)
N["S_E3U_COV_MAE"] = f(100 * (cov.E3u - cov.RG).abs().mean(), 2)
nc = 100 * (cov.E3_noclip - cov.RG).mean(); c999 = 100 * (cov.E3_c999 - cov.RG).mean()
N["S_NOCLIP_COV"] = f(nc, 2) if r(nc, 2) == r(c999, 2) else f"{f(nc, 2)} and {f(c999, 2)}"
N["S_C95_COV"] = f(100 * (cov.E3_c95 - cov.RG).mean(), 2)
TAB["synth_e3u"] = {scn: f(100 * (d.E3u - d.RG).mean(), 2) for scn, d in p1.groupby("scenario")}
p1 = p1.sort_values(["scenario", "rep", "t"])
p1["E0h"] = p1.groupby(["scenario", "rep"]).RG.shift(1)
TAB["synth_e0h"] = {scn: f(100 * (d.E0h - d.RG).mean(), 2) for scn, d in p1.groupby("scenario")}
cov = p1[p1.scenario == "covariate"]
_cc = cov[cov.t >= 3]
N["S_E3U_COV_COMMON"] = f(100 * (_cc.E3u - _cc.RG).mean(), 2)
N["S_E3_COV_COMMON"] = f(100 * (_cc.E3 - _cc.RG).mean(), 2)
N["S_ORACLE_COV_COMMON"] = f(100 * (_cc.E3_oracle - _cc.RG).mean(), 2)
N["S_E0H_COV"] = f(100 * (cov.E0h - cov.RG).mean(), 2)
N["S_E0H_COV_MAE"] = f(100 * (cov.E0h - cov.RG).abs().mean(), 2)

p2 = pd.read_csv(os.path.join(SYN2, "part2.csv"))
tr = p2.groupby(["rule", "t"]).RG.mean().mul(100)
for k, rule in [("FIX", "fixed_rule"), ("RT", "retrained"), ("RQ", "fixed_model_requantiled")]:
    N[f"P2_{k}_FIRST"] = f(tr[(rule, 2)], 2)
    N[f"P2_{k}_LAST"] = f(tr[(rule, 8)], 2)

p3 = pd.read_csv(os.path.join(SYN, "part3.csv"))
for e in EST:
    p3["ae_" + e] = 100 * (p3[e] - p3.RG).abs()
g = p3.groupby(["spec", "strength", "n_H"])[["ae_" + e for e in EST]].mean()
well, mis = g.loc["well-specified"], g.loc["misspecified"]
N["G_WELL_E2"], N["G_WELL_E3"] = f(well.ae_E2.mean(), 2), f(well.ae_E3.mean(), 2)
N["G_MIS_E1"] = f"{f(mis.ae_E1.min(), 1)}–{f(mis.ae_E1.max(), 1)}"
strong = mis.loc[1.6]
N["G_MIS_STRONG_GAIN"] = f((strong.ae_E2 - strong.ae_E3).mean(), 2)
N["G_C95_STRONG"] = f(mis.loc[(1.6, 20000)].ae_E3_c95, 2)
N["G_C99_STRONG"] = f(mis.loc[(1.6, 20000)].ae_E3, 2)
TAB["grid"] = []
for (spec, m, nh), row in g.iterrows():
    TAB["grid"].append([spec.replace("-", " ").capitalize(), f"{m:.1f}", f"{nh:,}"] +
                       [f"{r(row['ae_' + e], 2):.2f}" for e in ["E0", "E1", "E2", "E3u", "E3", "E3_c95", "E3_noclip", "E3_oracle"]])

# ------------------------------------------------------------------ hardware note
if len(sys.argv) > 5:
    a, b = pd.read_csv(sys.argv[4]), pd.read_csv(sys.argv[5])
    m = a.merge(b, on=["model", "year", "q"])
    m = m[(m.model == "xgb") & (m.q == Q)]
    N["XGB_HW_DIFF"] = f(100 * max((m.G_x - m.G_y).abs().max(), (m.RG_x - m.RG_y).abs().max()), 1)

json.dump(N, open("collected_numbers.json", "w"), ensure_ascii=False, indent=1)
json.dump(TAB, open("collected_tables.json", "w"), ensure_ascii=False, indent=1)
for k, v in N.items():
    print(f"{k:22s} {v}")
