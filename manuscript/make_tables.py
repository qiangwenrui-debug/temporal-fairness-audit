"""Builds every table's cells from results_data.py and writes tables.json."""
import json
import numpy as np
from results_data import (YEARLY, POOLED, PATTERN, ROBUST, SYNTH, SENS)

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
    return f"{num(a, nd)}–{num(b, nd)}" if num(a, nd) != num(b, nd) else num(a, nd)


T = {}
import os
# cells of Tables V-VIII, A3-A5 written by collect_results.py
REV = json.load(open("collected_tables.json"))

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

T["robust"]["rows"] += REV["robust_extra"]
# ---- Table VI: synthetic
srows = []
for scn, (g, rg, rc, bias, mae) in SYNTH.items():
    f = lambda a: f"{num(a[0], 1)} → {num(a[1], 1)}"
    bb = [num(b, 2) if b != 0 else "0.00" for b in bias]
    key = {"No drift": "none", "Covariate shift": "covariate", "Label shift": "label",
           "Concept shift": "concept"}[scn]
    srows.append([scn, f(g), f(rg), f(rc)] + [REV["synth_e0h"][key]] + bb[:3] + [REV["synth_e3u"][key]] + bb[3:])
T["synth"] = {"header": ["Scenario", "*G*_{t} first → last (pp)", "*RG*_{t} first → last (pp)",
                         "*RC*_{t} first → last (pp)", "Bias E0", "Bias E0′", "Bias E1", "Bias E2", "Bias E3u",
                         "Bias E3", "Bias E3, true ratio"], "rows": srows}

# ---- Table VII: monitoring error
mrows = REV["monitor"]
T["monitor"] = {"header": ["Estimator", "MAE (pp)", "Max error (pp)", "Error 2021 (pp)",
                           "Alerts raised", "Preci\u00adsion", "Recall", "MAE, all years (pp)"], "rows": mrows}

# ---- Table VIII: policies
T["policy"] = {"header": ["Policy", "Years outside ±5 pp", "Largest |*G*_{t}| (pp)", "Approval rate (%)",
                          "FRR (%)", "FAR (%)", "Largest |*RG*_{t}| (pp)"], "rows": REV["policy"]}

# ---- Table A1: tolerance sensitivity (XGBoost yearly gaps)
deltas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
arows = []
for key in ["csmar", "inc_race_ca", "inc_race_tx", "emp_race_ca", "inc_sex_ca"]:
    G = np.abs(np.array(YEARLY[key]["G"])) * 100
    arows.append([YEARLY[key]["label"]] + [f"{int((G > dl + 1e-9).sum())}/{len(G)}" for dl in deltas])
arows.append(["Employment × sex, CA"] + ["8/8"] + ["0/8"] * (len(deltas) - 1))  # yearly G +1.1 to +1.7
T["delta"] = {"header": ["Setting"] + [f"±{d}" for d in deltas], "rows": arows}

# ---- Table A2: timeline of the two protocols
T["timeline"] = {"header": ["Test year *T*", "Audit protocol: training years (*d* = 1)", "Monitoring: training years",
                            "Monitoring: reference year *H*", "Labels of *T* first used (*d* = 1)", "*d* = 2: training / reference"],
                 "rows": [["2015", "2014", "–", "–", "2016", "–"],
                          ["2016", "2014–2015", "2014", "2015", "2017", "–"],
                          ["2017", "2014–2016", "2014–2015", "2016", "2018", "2014 / 2015"],
                          ["2018", "2014–2017", "2014–2016", "2017", "2019", "2014–2015 / 2016"],
                          ["2019", "2014–2018", "2014–2017", "2018", "2021", "2014–2016 / 2017"],
                          ["2021", "2014–2019", "2014–2018", "2019", "2022", "2014–2018 / 2019"],
                          ["2022", "2014–2021", "2014–2019", "2021", "2023", "2014–2018 / 2019"],
                          ["2023", "2014–2022", "2014–2021", "2022", "after 2023", "2014–2019 / 2021"]]}
# ---- Table A3: pooled beta applied in every year, approval rates
T["bpool"] = {"header": ["Setting", "β^{*}_{pool}", "Years within ±5 pp, β = 0", "Years within ±5 pp, β^{*}_{pool}",
                         "*G*_{t} at β^{*}_{pool} (pp)", "Max |approval − *q*| at β^{*}_{t} (pp)",
                         "Max |approval − *q*| at β^{*}_{pool} (pp)", "Max |β^{*}_{t} − β^{*}_{t}, fixed budget|",
                         "Composition term (pp)"], "rows": REV["bpool"]}
# ---- Table A4: synthetic grid
T["grid"] = {"header": ["Scoring model", "Shift (sd)", "*n*_{H}", "E0′", "E1", "E2", "E3u", "E3",
                        "E3, clip 95%", "E3, no clip", "E3, true ratio"], "rows": REV["grid"]}
# ---- Table A5: other states
T["states"] = {"header": ["State", "Years |*RG*_{t}| > 2 pp", "MAE E0", "MAE E0′", "MAE E1", "MAE E2", "MAE E3u", "MAE E3",
                          "E3 alerts raised / correct", "Max error E3 (pp)", "Min ESS, *g* = 1, *y* = 1 (%)"],
               "rows": REV["states"]}
# ---- Table A7: corporate credit data dictionary
T["csmar"] = {"header": ["Item", "Definition"], "rows": [
    ["Source", "CSMAR consolidated annual statements of A-share firms (tables FS_Combas, FS_Comins, FS_Comscfd), fiscal years 2010–2024; firms later delisted are retained; financial firms (CSRC section J) excluded; 41,763 firm-years of 4,974 firms with feature years 2010–2022"],
    ["Financial distress (*y* = 0)", "Net profit negative in both *t* + 1 and *t* + 2, or shareholders’ equity negative at the end of *t* + 1 or *t* + 2, or the firm stops filing within the window and its last report shows a loss, negative equity or special-treatment (ST) status; exits with a healthy last report (mostly mergers) have no outcome"],
    ["Heavy-polluting (*g* = 1)", "Year-specific CSRC-2012 industry code (CSMAR FI_T1) in B06, B07, B08, B09, C17, C19, C22, C25, C26, C28, C29, C30, C31, C32, C33 or D44, following the 2008 directory of the Ministry of Environmental Protection"],
    ["Size, leverage, profitability", "ln(total assets); total liabilities / total assets; net profit / total assets (ROA)"],
    ["Operations", "Operating profit / total operating revenue; administrative expenses / total operating revenue"],
    ["Cash flows", "Net operating, investing and financing cash flow, each / total assets"],
    ["Balance-sheet items", "Cash, accounts receivable, inventories and financial expenses, each / total assets"],
    ["Growth", "Growth of total operating revenue and of total assets relative to the previous year"],
    ["Indicators", "Loss (current-year net profit < 0); negative equity (year-end equity < 0)"],
    ["Treatment", "Continuous ratios winsorized at the 1st and 99th percentiles within each year; missing ratios set to the yearly median with a missing-value indicator; the group indicator is not a model input"],
    ["Models", "Logistic regression (L2, *C* = 10, standardized); XGBoost (150 trees, learning rate 0.03, depth 2, subsampling 0.8); LightGBM (150 trees, learning rate 0.03, 7 leaves, subsampling 0.8); selected by log-loss on training years 2010–2011 and validation year 2012"],
    ["Timing", "Test year *T* uses year-*T* statements and a model trained on feature years ≤ *T* − 2, whose outcomes need statements up to year *T*; all are due by 30 April of *T* + 1, the date at which the screen is assumed to decide"]]}

# ---- Table A8: timing sensitivity of the corporate credit case
import pandas as _pd
_tc = _pd.read_csv("timing_check.csv")
_dn = {"base": "Original: window *t* + 1, *t* + 2; *d* = 2", "lag3": "Training labels one year older: *d* = 3",
       "post": "Window *t* + 2, *t* + 3 (after decision); *d* = 3",
       "disc": "Original, with disclosure filter", "post_disc": "Window *t* + 2, *t* + 3, with disclosure filter"}
_mn = {"LR": "LR", "XGB": "XGBoost", "LGBM": "LightGBM"}
_rows = []
for dsg in ["base", "disc", "lag3", "post", "post_disc"]:
    for i, (_, x) in enumerate(_tc[_tc.design == dsg].iterrows()):
        _rows.append([_dn[dsg] if i == 0 else "", x.test_years.replace("-", "–") if i == 0 else "", _mn[x.model],
                      f"{pp(x.pooled)} [{pp(x.lo)}, {pp(x.hi)}]", f"{r(100 * x.mean_abs, 1):.1f}", f"{x.out5}/{x.n_years}",
                      rng(x.beta_min, x.beta_max), num(x.beta_pool), f"{x.carry_out}/{x.carry_n}",
                      x.good_2014_16.replace("-", M).replace(" to ", " to ")])
T["timing"] = {"header": ["Design", "Test years", "Model", "Pooled gap [95% CI] (pp)", "Mean |*G*_{t}| (pp)",
                          "Years outside ±5 pp", "β^{*}_{t} range", "β^{*}_{pool}", "Carry-over: years outside",
                          "Gap among non-distressed, 2014–2016 (pp)"], "rows": _rows}

# ---- Table A6: approval share q (XGBoost, audit protocol; main runs)
_qs = [("results_income_race_CA_d1_quantile", "Income × race, CA"), ("results_income_race_TX-NY-FL_d1_quantile", "Income × race, TX/NY/FL"),
       ("results_employment_race_CA_d1", "Employment × race, CA"), ("results_income_sex_CA_d1_quantile", "Income × sex, CA"),
       ("results_employment_sex_CA_d1", "Employment × sex, CA")]
_rows = []
for fn, lab in _qs:
    _p = _pd.read_csv(f"qsens/{fn}.csv"); _p = _p[_p.model == "xgb"].sort_values("q")
    for i, (_, x) in enumerate(_p.iterrows()):
        _rows.append([lab if i == 0 else "", f"{x.q:.2f}", pp(x.pooled_gap), f"{r(100 * x.mean_abs_yearly, 1):.1f}",
                      f"{int(x.yrs_out_5)}/{int(x.n_years)}", rng(x.beta_min, x.beta_max, 3)])
T["qsens"] = {"header": ["Setting", "*q*", "Pooled gap (pp)", "Mean |*G*_{t}| (pp)", "Years outside ±5 pp", "β^{*}_{t} range"],
              "rows": _rows}

srows = []
for name, (hist, g11, g125, gmax) in SENS.items():
    c, n, w = hist
    hc = f"{c}/{n}" if n else f"{r(c, 2):.2f}"
    srows.append([name, hc, f"{w:.2f}", f"{g11[0]:.2f}", f"{g11[1]:.2f}", f"{g125[0]:.2f}", f"{g125[1]:.2f}",
                  "\u2013" if gmax is None else f"{gmax:.3f}"])
T["sens"] = {"header": ["Setting", "History \u0393: coverage", "History \u0393: width (pp)",
                        "\u0393 = 1.1: coverage", "\u0393 = 1.1: width (pp)", "\u0393 = 1.25: coverage",
                        "\u0393 = 1.25: width (pp)", "Largest \u0393 needed"], "rows": srows}
json.dump(T, open("tables.json", "w"), ensure_ascii=False, indent=1)
for k, v in T.items():
    print(k, len(v["rows"]), "rows")
print(T["delta"]["rows"])
