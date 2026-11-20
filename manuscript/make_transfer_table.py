"""Render the 2024 probability-scale policy comparison from its four-state CSV."""
import argparse
import csv
import math
from pathlib import Path


def render(source, output, uncertainty=None):
    uncertainty = uncertainty or source.parents[1] / "matched" / "frozen_2024_household_intervals.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["scale"] == "probability" and r["budget"] == "interpolated"]
    if [r["state"] for r in rows] != ["CA", "TX", "NY", "FL"]:
        raise ValueError("Expected exactly four unique state panels in the summary")
    with uncertainty.open(encoding="utf-8-sig", newline="") as stream:
        bootstrap = list(csv.DictReader(stream))
    ci = {(r["state"],r["policy"]):r for r in bootstrap}
    states = [r["state"] for r in rows]
    if len(bootstrap)!=8 or set(ci)!={(state,policy) for state in states for policy in ("pooled","common")}:
        raise ValueError("Expected both frozen-rule intervals in each of four states")
    policy_names = ("pooled", "common", "adaptive")
    tally = dict.fromkeys(policy_names, 0)
    text = [
        r"\begin{table*}[!t]",
        r"\caption{Black--White Approval Gaps in the 2024 Income Screen Under Historical and Current-Cohort Calibration}\label{tab:transfer}",
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{9}{r}@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Prior years pooled} & \multicolumn{3}{c}{Each prior year} & \multicolumn{3}{c}{Current-batch minimum}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"State & $\beta$ & $G$ (pp) & Changed (\%) & $\beta$ & $G$ (pp) & Changed (\%) & $\beta$ & $G$ (pp) & Changed (\%)\\",
        r"\midrule",
    ]
    for r in rows:
        cells = [r["state"]]
        for name in policy_names:
            b = float(r[name + "_2024_beta"])
            gap = float(r[name + "_2024_gap_pp"])
            changed = float(r[name + "_2024_changed_share_pp"])
            if not all(map(math.isfinite,(b,gap,changed))) or r[name + "_2024_within"] not in {"True","False"}:
                raise ValueError("Incomplete transfer result")
            within = r[name + "_2024_within"] == "True"
            if within != (abs(gap) <= 5):
                raise ValueError("Gap and stored verdict disagree")
            tally[name] += int(within)
            if name != "adaptive":
                c=ci[(r["state"],name)]
                if (c["year"]!="2024" or c["draws"]!="1000" or
                    not math.isclose(float(c["gap_pp"]),gap,abs_tol=1e-7) or
                    not math.isclose(float(c["beta"]),b,abs_tol=1e-9) or
                    not all(math.isfinite(float(c[k])) for k in ("lower_pp","upper_pp")) or
                    float(c["lower_pp"])>float(c["upper_pp"])):
                    raise ValueError("Bootstrap interval and transfer estimate do not match")
            value = f"{gap:.3f}"
            cells += [f"{b:.3f}", ("$" + value + "$") if within else r"$\mathbf{" + value + "}$", f"{changed:.3f}"]
        text.append(" & ".join(cells) + r"\\")
    text += [r"\midrule",
             "States within tolerance & " + " & ".join(r"\multicolumn{3}{c}{" + f"{tally[k]}/4" + "}" for k in policy_names) + r"\\",
             r"\bottomrule\end{tabular*}",
             r"\par\smallskip\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
             r"\toprule Pointwise 95\% interval for $G$ (pp) & California & Texas & New York & Florida\\\midrule"]
    for name in ("pooled","common"):
        intervals = []
        for state in states:
            c=ci[state,name]
            intervals.append("$[%.2f, %.2f]$" % (float(c["lower_pp"]),float(c["upper_pp"])))
        label = "Prior years pooled" if name == "pooled" else "Each prior year"
        text.append(label+" & "+" & ".join(intervals)+r"\\")
    text += [r"\bottomrule\end{tabular*}",
             r"\par\vspace{3pt}\begin{minipage}{\textwidth}\footnotesize $G$ is the Black approval rate minus the White approval rate at a 75\% approval target. Each state has a separate XGBoost model (seed 0), trained on 2014--2019 and 2021--2023. The historical minima use eight audit cohorts in 2015--2019 and 2021--2023. The pooled constraint applies to all cohorts combined and the year-by-year constraint to each cohort. The current-batch minimum uses 2024 scores and group memberships, so its count measures within-batch feasibility. Bold point estimates exceed the $\pm5$ pp band. Changed is the share of 2024 respondents whose simulated decision differs from the unadjusted rule. For rules chosen from earlier years, 1,000 household-cluster bootstrap samples per state hold model and strength fixed and recompute the 2024 score quantiles. The pointwise intervals condition on those fits.\end{minipage}",
             r"\end{table*}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/transfer/transfer_state_summary.csv"))
    parser.add_argument("--uncertainty", type=Path, help="matched 2024 household-bootstrap CSV")
    parser.add_argument("--out", type=Path, default=Path("paper/generated/transfer.tex"))
    args = parser.parse_args()
    render(args.source, args.out, args.uncertainty)
