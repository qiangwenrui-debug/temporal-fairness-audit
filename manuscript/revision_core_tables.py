"""Build the three main-text comparison tables from machine-readable results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METHODS = ["E0", "E1", "E2", "E3u", "E3"]
RULES = {"none": "Uncalibrated", "per_period": "Annual minimum",
         "per_period_budget": "Annual, fixed budget"}
WINDOWS = {"Historical": [2017, 2018, 2019, 2021, 2022, 2023], "2024": [2024]}


def table(path, title, label, columns, header, body, note):
    text = "\n".join([
        r"\begin{table*}[!t]",
        rf"\caption{{{title}}}\label{{{label}}}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{2.4pt}",
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{columns}@{{}}}}",
        r"\toprule", header + r" \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular*}", r"\par\vspace{3pt}",
        r"\begin{minipage}{\textwidth}\footnotesize " + note + r"\end{minipage}",
        r"\end{table*}", "",
    ])
    path.write_text(text, encoding="utf-8")


def selection(frame, years, rule):
    subset = frame.loc[
        frame.state.isin(["CA", "TX", "NY", "FL"]) & frame.task.eq("income")
        & frame.group.eq("race") & frame.label.eq("quantile") & frame.delay.eq(1)
        & frame.seed.eq(0) & frame.year.isin(years) & frame.policy.eq(rule)
    ].copy()
    if len(subset) != 4*len(years) or subset.duplicated(["state", "year"]).any():
        raise ValueError(f"Missing or duplicate state-year for {rule}, {years}")
    return subset


def row(fields):
    return " & ".join(map(str, fields)) + r" \\"


def observed_fraction(count, eligible):
    """An empty opportunity set has no empirical rate."""
    return f"{count}/{eligible}" if eligible else r"---"


def build(results, out):
    out.mkdir(parents=True, exist_ok=True)
    monitor = results / "monitor"
    yearly = pd.read_csv(monitor / "all_yearly.csv")
    intervals = pd.read_csv(monitor / "all_intervals.csv")
    point_rows, interval_rows = [], []
    for window, years in WINDOWS.items():
        if point_rows:
            point_rows.append(r"\midrule")
            interval_rows.append(r"\midrule")
        for j, (rule, label) in enumerate(RULES.items()):
            s = selection(yearly, years, rule)
            if not np.isfinite(s[["true_RG", *METHODS]].to_numpy()).all():
                raise ValueError("The common estimator window contains missing values.")
            actual = s.true_RG.abs() > .02
            alarm = s.E3.abs() > .02
            mae = s[METHODS].sub(s.true_RG, axis=0).abs().mean()*100
            point_rows.append(row([window if j == 0 else "", label, *[f"{m:.2f}" for m in mae],
                                   f"{s.true_RG.abs().max()*100:.2f}", f"{actual.sum()}/{len(s)}",
                                   observed_fraction((alarm & ~actual).sum(), (~actual).sum()),
                                   observed_fraction((~alarm & actual).sum(), actual.sum())]))
        configurations = [(rule, "history") for rule in RULES]
        if window == "Historical":
            configurations.append(("per_period", "fixed_1.25"))
        for j, (rule, method) in enumerate(configurations):
            s = selection(intervals[intervals.gamma_method.eq(method)], years, rule)
            if not np.isfinite(s[["lo_pp","hi_pp","true_RG_pp"]].to_numpy()).all() or (s.lo_pp > s.hi_pp).any():
                raise ValueError("Missing or invalid sensitivity interval.")
            out_band = (s.lo_pp > 2) | (s.hi_pp < -2)
            in_band = (s.lo_pp >= -2) & (s.hi_pp <= 2)
            covered = (s.lo_pp <= s.true_RG_pp) & (s.true_RG_pp <= s.hi_pp)
            truth_out = s.true_RG_pp.abs() > 2
            wrong = (out_band & ~truth_out) | (in_band & truth_out)
            unresolved = ~(in_band | out_band)
            interval_rows.append(row([window if j == 0 else "", RULES[rule],
                                      "History" if method == "history" else "1.25",
                                      f"{covered.sum()}/{len(s)}", f"{(s.hi_pp-s.lo_pp).mean():.2f}",
                                      out_band.sum(), in_band.sum(),
                                      f"{unresolved.sum()} ({(unresolved & truth_out).sum()})",
                                      observed_fraction(wrong.sum(), (~unresolved).sum())]))
    table(out/"point.tex", "Estimation Errors and Alerts for the Black--White Approval Gap Among Higher-Income Respondents",
          "tab:VII", "llrrrrrrrrr",
          r"& & \multicolumn{5}{c}{Mean absolute error (pp)} & & & \multicolumn{2}{c}{Weighted reference} \\"
          r"\cmidrule(lr){3-7}\cmidrule(l){10-11} "
          r"Window & Decision rule & \shortstack{Last\\observed} & \shortstack{Raw\\scores} & \shortstack{Recalibrated\\scores} & \shortstack{Unweighted\\reference} & \shortstack{Weighted\\reference} & Max.\ $|RG_t|$ & Events/$n$ & False alerts & Misses",
          point_rows, r"$RG_t$ is the Black approval rate minus the White approval rate among respondents above the state's annual income cutoff. An event is an observed $|RG_t|>2$ percentage points (pp). The last two columns count false alerts among non-events and misses among events. A dash marks an empty denominator. Both reference estimators apply the current rule to the latest labelled cohort. California, Texas, New York and Florida contribute six historical years each (2017--2019 and 2021--2023) and one test each in 2024. The target approval share is 75\% and labels arrive one year later.")
    table(out/"interval.tex", "Coverage and Alert Decisions for Residual Sensitivity Intervals by Decision Rule",
          "tab:IX", "lllrrrrrr",
          r"Window & Decision rule & $\Gamma$ & Covered & Mean width (pp) & Outside & Inside & Unresolved (events) & Wrong/definite",
          interval_rows, r"The residual is the Black minus White approval rate among respondents above the state's annual income cutoff, at a 75\% target approval share. Covered counts intervals containing the observed residual. Outside and Inside count intervals lying wholly outside or inside the $\pm2$ percentage-point band. For intervals crossing a boundary, parentheses count observed breaches. Wrong/definite counts incorrect classifications among all resolved cases. History uses the largest sensitivity factor $\Gamma$ needed to cover an earlier observed residual under that rule. Each of the four states contributes six historical tests (2017--2019 and 2021--2023) and one in 2024.")
    sim = pd.read_csv(results/"matched"/"simulation_paired.csv").set_index("scenario")
    if not sim.index.is_unique or not np.isfinite(sim.to_numpy(dtype=float)).all():
        raise ValueError("Duplicate scenarios or nonfinite simulation errors.")
    sim_rows = []
    def delta(s, base):
        def signed(val):
            digits = 4 if 0 < abs(val) < .0005 else 3
            return f"{val:+.{digits}f}"
        return ("$" + signed(s["E3_minus_"+base]) + r"\;[" + signed(s[base+"_lo"])
                + "," + signed(s[base+"_hi"]) + "]$")
    for key, label in [("none","No shift"),("covariate","Covariate"),("label","Label"),("concept","Concept")]:
        s = sim.loc[key]
        mae = [s[k] for k in ["E0","E0prime","E1","E2","E3u","E3","E3_oracle"]]
        sim_rows.append(row([label,*[f"{v:.2f}" for v in mae],delta(s,"E3u"),delta(s,"E2")]))
    table(out/"simulation.tex", "Residual Estimation Errors and Paired Differences in the Simulations",
          "tab:VI", "lrrrrrrrcc",
          r" & \multicolumn{7}{c}{Mean absolute error (pp)} & \multicolumn{2}{c}{Weighted reference minus comparator (pp)} \\"
          r"\cmidrule(lr){2-8}\cmidrule(l){9-10} "
          r"Shift & \shortstack{Last\\observed} & \shortstack{Reference\\thresholds} & \shortstack{Raw\\scores} & \shortstack{Recalibrated\\scores} & \shortstack{Unweighted\\reference} & \shortstack{Weighted\\reference} & \shortstack{True density\\ratio} & \shortstack{Unweighted reference\\(95\% interval)} & \shortstack{Recalibrated scores\\(95\% interval)}",
          sim_rows, r"The target is the group 1 minus group 0 approval rate among simulated individuals with favourable outcomes. Errors in percentage points are averaged over periods 3--8 in 20 independent replicates per scenario. The reference-threshold estimator uses the current model and thresholds calculated on the labelled reference cohort. The unweighted and weighted reference estimators use the current thresholds on that cohort. Weighting uses the estimated covariate density ratio except in the true-density-ratio column. Paired differences use 10,000 bootstrap samples of complete replicates. Negative values favour the weighted reference estimator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    parser.add_argument("--out", type=Path, default=ROOT/"paper"/"generated")
    args = parser.parse_args()
    build(args.results, args.out)
