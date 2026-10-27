"""Builds every table's cells from results_data.py and writes tables.json for the Word builder."""
import json
import numpy as np
from results_data import (YEARLY, POOLED, PATTERN, ROBUST, MON_ERR, POLICY, SYNTH)

M = "−"  # minus sign


from decimal import Decimal, ROUND_HALF_UP


def r(x, nd):
    """round half away from zero on the decimal representation (0.245 -> 0.25)."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pp(x, nd=1):
    """fraction -> signed percentage points string with a true minus sign."""
    v = r(100 * x, nd)
    s = f"{abs(v):.{nd}f}"
    return (M + s) if v < 0 else ("+" + s if v > 0 else s)


def num(x, nd=2):
    v = r(x, nd)
    s = f"{abs(v):.{nd}f}"
    return (M + s) if v < 0 else s


def rng(a, b, nd=2):
    return f"{num(a, nd)}–{num(b, nd)}" if abs(a - b) > 1e-9 else num(a, nd)


T = {}

# ---- Table III: settings
T["settings"] = {
    "header": ["Setting", "Favourable outcome", "Group 1 / group 0", "Test years", "d",
               "Favourable rate, group 1 (%)", "Favourable rate, group 0 (%)", "Individuals per test year"],
    "rows": [
        ["Corporate credit (CSMAR)", "No financial distress within two years",
         "Heavy-polluting / other industries", "2014–2022", "2", "84.7–96.5", "80.5–97.3",
         "2,586–4,812 (group 1: 684–1,042)"],
    ]}
NPER = {"inc_race_ca": "88,132–129,636 (group 1: 7,963–8,691)",
        "inc_race_tx": "226,617–298,737 (group 1: 32,214–36,549)",
        "emp_race_ca": "178,282–253,593 (group 1: 17,181–19,370)",
        "inc_sex_ca": "187,475–203,278 (group 1: 87,957–96,150)"}
for key, outcome, groups in [("inc_race_ca", "Income in top 40% of the year", "Black / White"),
                             ("inc_race_tx", "Income in top 40% of the year", "Black / White"),
                             ("emp_race_ca", "Employed", "Black / White"),
                             ("inc_sex_ca", "Income in top 40% of the year", "Female / male")]:
    d = YEARLY[key]
    T["settings"]["rows"].append([
        d["label"], outcome, groups, "2015–2019, 2021–2023", "1",
        f"{100*min(d['pi1']):.1f}–{100*max(d['pi1']):.1f}",
        f"{100*min(d['pi0']):.1f}–{100*max(d['pi0']):.1f}", NPER[key]])
T["settings"]["rows"].append(["Employment × sex, CA", "Employed", "Female / male",
                              "2015–2019, 2021–2023", "1", "40.4–42.9", "47.2–50.0",
                              "374,943–392,318 (group 1: 190,306–198,664)"])

# ---- Table IV: summary across settings and models
rows = []
for key in ["csmar", "inc_race_ca", "inc_race_tx", "emp_race_ca", "inc_sex_ca"]:
    for i, model in enumerate(["LR", "XGB", "LGBM"]):
        g, lo, hi, mabs, out, n, bmin, bmax, bpool, cf, cn, cmax = POOLED[key][model]
        rows.append([YEARLY[key]["label"] if i == 0 else "", PATTERN[key] if i == 0 else "",
                     {"LR": "LR", "XGB": "XGBoost", "LGBM": "LightGBM"}[model],
                     f"{pp(g)} [{pp(lo)}, {pp(hi)}]", f"{r(100*mabs, 1):.1f}", f"{out}/{n}",
                     rng(bmin, bmax), num(bpool), f"{cf}/{cn} ({r(100*cmax, 1):.1f})" if cf else f"{cf}/{cn} (–)"])
for i, (mdl, g, lo, hi) in enumerate([("LR", "+2.2", "+2.1", "+2.3"), ("XGBoost", "+1.4", "+1.3", "+1.5"),
                                      ("LightGBM", "+1.4", "+1.3", "+1.5")]):
    rows.append(["Employment × sex, CA" if i == 0 else "", "Stable (control)" if i == 0 else "", mdl,
                 f"{g} [{lo}, {hi}]", g.lstrip("+"), "0/8", "0.00", "0.00", "0/7 (–)"])
T["summary"] = {"header": ["Setting", "Pattern", "Model", "Pooled gap [95% CI] (pp)",
                           "Mean |*G*_{t}| (pp)", "Years outside ±5 pp", "β^{*}_{t} range",
                           "β^{*}_{pool}", "Carry-over: years outside (largest |gap|, pp)"],
                "rows": rows}

# ---- Table V: robustness
T["robust"] = {"header": ["Income label", "d", "Mean gap 2015\u20132019 (pp)", "Mean gap 2021–2023 (pp)",
                          "Pooled gap (pp)", "β^{*}_{t} range", "β^{*}_{pool}"],
               "rows": [[lab, str(d), pp(pre) + (" (2016–19)" if d == 2 else ""), pp(post), pp(pool),
                         rng(0, bmax), num(bpool)] for lab, d, pre, post, pool, bmax, bpool in ROBUST]}

T["robust"]["rows"] += [
    ["Top 40% within Black and White (check baseline)", "1", "−3.5", "−11.7", "−6.5", "0.00–0.64", "–"],
    ["  with survey weights (PWGTP)", "1", "−1.1", "−8.9", "−3.9", "0.00–0.49", "–"],
    ["  without occupation (OCCP)", "1", "−2.8", "−9.9", "−5.4", "0.00–0.54", "–"]]
# ---- Table VI: synthetic
srows = []
for scn, (g, rg, rc, bias, mae) in SYNTH.items():
    f = lambda a: f"{num(a[0], 1)} → {num(a[1], 1)}"
    srows.append([scn, f(g), f(rg), f(rc)] + [num(b, 2) if b != 0 else "0.00" for b in bias])
T["synth"] = {"header": ["Scenario", "*G*_{t} first → last (pp)", "*RG*_{t} first → last (pp)",
                         "*RC*_{t} first → last (pp)", "Bias E0", "Bias E1", "Bias E2", "Bias E3",
                         "Bias E3, true ratio"], "rows": srows}

# ---- Table VII: monitoring error
mrows = []
for (setting, true_alerts), ests in MON_ERR.items():
    mrows.append({"group": f"{setting} (years with |*RG*_{{t}}| > 2 pp: {true_alerts})"})
    for e, mae, mx, e21, raised, prec, rec in ests:
        mrows.append([e, f"{mae:.2f}", f"{mx:.2f}", f"{e21:.2f}", str(raised),
                      "–" if prec is None else f"{prec:.2f}", "–" if rec is None else f"{rec:.2f}"])
T["monitor"] = {"header": ["Estimator", "MAE (pp)", "Max error (pp)", "Error 2021 (pp)",
                           "Alerts raised", "Preci\u00adsion", "Recall"], "rows": mrows}

# ---- Table VIII: policies
prow = []
for setting, pols in POLICY.items():
    prow.append({"group": setting})
    for pol, (out, n, mx, frr) in pols.items():
        prow.append([pol, f"{out}/{n}", f"{mx:.1f}", f"{frr:.2f}"])
T["policy"] = {"header": ["Policy", "Years outside ±5 pp", "Largest |*G*_{t}| (pp)",
                          "False-rejection rate (%)"], "rows": prow}

# ---- Table A1: tolerance sensitivity (XGBoost yearly gaps)
deltas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
arows = []
for key in ["csmar", "inc_race_ca", "inc_race_tx", "emp_race_ca", "inc_sex_ca"]:
    G = np.abs(np.array(YEARLY[key]["G"])) * 100
    arows.append([YEARLY[key]["label"]] + [f"{int((G > dl + 1e-9).sum())}/{len(G)}" for dl in deltas])
arows.append(["Employment × sex, CA"] + ["8/8"] + ["0/8"] * (len(deltas) - 1))  # yearly G +1.1 to +1.7
T["delta"] = {"header": ["Setting"] + [f"±{d}" for d in deltas], "rows": arows}

json.dump(T, open("tables.json", "w"), ensure_ascii=False, indent=1)
for k, v in T.items():
    print(k, len(v["rows"]), "rows")
print(T["delta"]["rows"])
