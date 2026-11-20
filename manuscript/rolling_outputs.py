"""Draw the rolling transfer figure and typeset the full fold results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from revision_figures import BAND, BLUE, INK, MUTED, ORANGE, STATES, WIDTH


ROOT = Path(__file__).resolve().parents[2]
STATE_NAMES = {"CA": "California", "TX": "Texas", "NY": "New York", "FL": "Florida"}
POLICIES = {
    "pooled": ("Prior years pooled", BLUE, "o", "-", -.12),
    "common": ("Each prior year", ORANGE, "s", "--", .12),
}
TOLERANCE = 5.


def load_folds(path, origin):
    frame = pd.read_csv(path)
    if set(frame.state) != set(STATES) or set(frame.policy) != set(POLICIES):
        raise ValueError("Expected both historical policies in all four states.")
    years = sorted(frame.year.unique())
    if 2020 in years or years != [y for y in range(years[0], years[-1]+1) if y != 2020]:
        raise ValueError("Check the standard-product year coverage.")
    folds = frame.set_index(["state", "year", "policy"]).sort_index()
    grid = pd.MultiIndex.from_product([STATES, years, POLICIES], names=folds.index.names)
    if not folds.index.is_unique or len(grid.difference(folds.index)):
        raise ValueError("A state, test year or frozen rule is missing or repeated.")
    numeric = folds[["beta", "gap_pp", "lower_pp", "upper_pp", "changed_share_pp", "annual_beta", "n"]]
    if not np.isfinite(numeric).all().all() or (folds.lower_pp > folds.upper_pp).any():
        raise ValueError("Missing fold result or invalid interval.")
    if not (folds.within == (folds.gap_pp.abs() <= TOLERANCE)).all():
        raise ValueError("Recorded classifications disagree with the full-precision gap.")
    for _, pair in folds.groupby(level=["state", "year"]):
        if pair.n.nunique() != 1 or pair.annual_beta.nunique() != 1 or not pair.q.eq(.75).all():
            raise ValueError("The two rules must be applied to the same cohort and budget.")
    if years[0] != origin+2:
        raise ValueError("The first audited fold must follow the initial model and calibration cohort.")
    for year in years:
        prior = [y for y in range(origin, year) if y != 2020]
        rows = folds.xs(year, level="year")
        if not rows.train_years.eq("|".join(map(str, prior))).all():
            raise ValueError(f"Unexpected model training window for {year}.")
        if not rows.history_years.astype(str).eq("|".join(map(str, prior[1:]))).all():
            raise ValueError(f"Unexpected history for strength selection in {year}.")
    return folds, years


def draw_figure(folds, years, directory, stem):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.00), sharey=True, sharex=True)
    fig.subplots_adjust(left=.091, right=.986, top=.93, bottom=.195, wspace=.13, hspace=.36)
    full_axis = np.arange(years[0], years[-1]+1)
    lower = 2*np.floor((min(folds.lower_pp.min(), -TOLERANCE)-.25)/2)
    upper = 2*np.ceil((max(folds.upper_pp.max(), TOLERANCE)+.25)/2)
    edges = [edge for edge in (-TOLERANCE, TOLERANCE) if lower < edge < upper]
    ticks = [tick for tick in np.arange(lower, upper+.1, 2)
             if all(abs(tick-edge) > 1 for edge in edges)]
    for index, (ax, state) in enumerate(zip(axes.flat, STATES)):
        ax.axhspan(-TOLERANCE, TOLERANCE, color=BAND, lw=0, zorder=0)
        for edge in edges:
            ax.axhline(edge, ls=(0, (2.5, 2)), lw=.85, color=INK, zorder=1)
        if 2020 in full_axis:
            ax.axvspan(2019.72, 2020.28, color="white", alpha=.82, lw=0, zorder=.5)
        for policy, (_, color, marker, style, offset) in POLICIES.items():
            subset = folds.xs((state, policy), level=["state", "policy"]).reindex(full_axis)
            x = full_axis+offset
            y = subset.gap_pp.to_numpy()
            ax.plot(x, y, lw=1.05, ls=style, color=color, zorder=2)
            # Draw interval endpoints directly. A percentile interval need not contain its point estimate.
            ax.vlines(x, subset.lower_pp, subset.upper_pp, lw=.72, color=color, alpha=.72, zorder=2)
            for endpoint in (subset.lower_pp, subset.upper_pp):
                ax.hlines(endpoint, x-.073, x+.073, lw=.72, color=color, alpha=.72, zorder=2)
            for is_inside in (True, False):
                subset_mask = subset.within.eq(is_inside).to_numpy()
                ax.plot(x[subset_mask], y[subset_mask], ls="none", marker=marker, ms=3.8,
                        markeredgewidth=.85, markeredgecolor=color,
                        markerfacecolor="white" if is_inside else color, zorder=3)
        ax.set_title(f"({chr(97+index)}) {STATE_NAMES[state]}", loc="left", fontsize=8.3, pad=4)
        ax.set_xlim(years[0]-.48, years[-1]+.48)
        ax.set_ylim(lower, upper)
        ax.set_yticks(sorted(ticks+edges))
        year_labels = [f"’{y % 100:02}" if len(full_axis)>10 else str(y) for y in full_axis]
        ax.set_xticks(full_axis, year_labels, fontsize=6.6)
        if index % 2 == 0:
            ax.set_ylabel("Approval gap (pp)", fontsize=7.5)
    handles = [
        Line2D([], [], label=label, color=color, lw=1.1, ls=style, marker=marker, ms=4,
               markerfacecolor="white") for label, color, marker, style, _ in POLICIES.values()
    ]
    for filled, label in [(True, "Outside ±5 pp"), (False, "Within ±5 pp")]:
        handles.append(Line2D([], [], label=label, ls="none", marker="o", ms=4,
                              markeredgecolor=INK, markerfacecolor=INK if filled else "white"))
    fig.text(.54, .103, "Test year", ha="center", va="center", color=MUTED, fontsize=7.4)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .012), ncol=4,
               frameon=False, handlelength=2, columnspacing=1.25)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/(stem+".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory/(stem+".png"), dpi=420)
    plt.close(fig)


def gap_cell(row):
    gap = f"{row.gap_pp:.3f}"
    if not row.within:
        return r"$\mathbf{" + gap + "}$"
    return f"${gap}$"


def interval_endpoint(value):
    # Preserve which side of the tolerance a near-boundary endpoint falls on.
    places = 2
    while abs(float(f"{value:.{places}f}")) == TOLERANCE and abs(value) != TOLERANCE and places < 8:
        places += 1
    return f"{value:.{places}f}"


def detailed_table(folds, years, states, label):
    region_names = " and ".join(STATE_NAMES[state] for state in states)
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Rolling Income Audits in " + region_names
        + r" With Training From 2010 ($q=0.75$)}\label{" + label + "}",
        r"\centering\footnotesize\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llr rrrr rrrr r@{}}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{Prior years pooled} & \multicolumn{4}{c}{Each prior year} & Current batch \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(l){12-12}",
        r"State & Year & People & $\beta$ & $G$ (pp) & 95\% interval & Changed (\%) & $\beta$ & $G$ (pp) & 95\% interval & Changed (\%) & $\beta$\\",
        r"\midrule",
    ]
    for index, state in enumerate(states):
        if index:
            lines.append(r"\midrule")
        for year in years:
            pooled = folds.loc[(state, year, "pooled")]
            common = folds.loc[(state, year, "common")]
            cells = [state if year == years[0] else "", str(year), f"{pooled.n:,.0f}"]
            for row in (pooled, common):
                interval = f"$[{interval_endpoint(row.lower_pp)}, {interval_endpoint(row.upper_pp)}]$"
                changed = "0" if row.changed_share_pp == 0 else f"{row.changed_share_pp:.2f}"
                if row.changed_share_pp > 0 and changed == "0.00":
                    changed = r"$<0.005$"
                cells.extend([f"{row.beta:.3f}", gap_cell(row), interval, changed])
            cells.append(f"{pooled.annual_beta:.3f}")
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular*}\par\smallskip",
        r"\begin{minipage}{.99\textwidth}\footnotesize "
        r"$G$ is the Black approval rate minus the White approval rate. Bold values fall outside $\pm5$ pp. "
        r"Each historical rule fixes the adjustment strength $\beta$ before the test year. Changed is the share whose decision differs from the unadjusted rule on the same scores. "
        r"The final column gives the smallest $\beta$ that meets the tolerance on the test batch itself. "
        r"Pointwise intervals resample households within the test cohort and recompute the score thresholds at the frozen strength."
        r"\end{minipage}\end{table*}",
    ])
    return lines


def original_window_table(folds, years):
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Rolling Income Audits With Training From 2014 ($q=0.75$)}\label{tab:rolling2014}",
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{.92\textwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}\toprule",
    ]
    for states in [STATES[:2], STATES[2:]]:
        left, right = (STATE_NAMES[state] for state in states)
        lines.append(f" & \\multicolumn{{4}}{{c}}{{{left}}} & \\multicolumn{{4}}{{c}}{{{right}}}" + r" \\")
        lines.extend([
            r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
            r" & \multicolumn{2}{c}{Prior years pooled} & \multicolumn{2}{c}{Each prior year}"
            r" & \multicolumn{2}{c}{Prior years pooled} & \multicolumn{2}{c}{Each prior year}\\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
            r"Year & $\beta$ & $G$ (pp) & $\beta$ & $G$ (pp) & $\beta$ & $G$ (pp) & $\beta$ & $G$ (pp)\\\midrule",
        ])
        for year in years:
            cells = [str(year)]
            for state in states:
                for policy in POLICIES:
                    row = folds.loc[(state, year, policy)]
                    cells.extend([f"{row.beta:.3f}", gap_cell(row)])
            lines.append(" & ".join(cells) + r" \\")
        if states[0] == STATES[0]:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule\end{tabular*}\par\smallskip",
        r"\begin{minipage}{.92\textwidth}\footnotesize For each test year the pooled rule "
        r"combines the earlier audit cohorts when choosing $\beta$. The year-by-year rule requires each earlier "
        r"cohort to meet the tolerance. $G$ is the test-year Black approval rate minus the White approval rate, "
        r"in percentage points. Bold values fall outside $\pm5$ pp.\end{minipage}\end{table*}",
    ])
    return lines


def summary_table(folds, previous_path):
    previous = pd.read_csv(previous_path).set_index(["state", "year", "policy"])
    pooled = folds.xs("pooled", level="policy")
    last = previous.droplevel("policy").sort_index()
    if not last.index.equals(pooled.index) or set(previous.index.get_level_values("policy")) != {"previous"}:
        raise ValueError("The previous-audit rule must cover the same test cohorts as the frozen policies.")
    if not last.n.equals(pooled.n) or not last.within.eq(last.gap_pp.abs() <= TOLERANCE).all():
        raise ValueError("Check previous-audit sample sizes and tolerance classifications.")
    data = pd.concat([folds, previous])
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Rolling Tests of Historical Threshold Rules for the Black--White Income Screen}\label{tab:rollsummary}",
        r"\centering\footnotesize\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule",
        r"& \multicolumn{2}{c}{Pooled} & \multicolumn{2}{c}{Previous year} & \multicolumn{2}{c}{Each prior year}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"State & Within & Mean $|G|$ & Within & Mean $|G|$ & Within & Mean $|G|$\\\midrule",
    ]
    for state in [*STATES, "All states"]:
        combined = state == "All states"
        if combined:
            lines.append(r"\midrule")
        subset = data if combined else data.xs(state, level="state")
        cells = [state if combined else STATE_NAMES[state]]
        for policy in ("pooled", "previous", "common"):
            group = subset.xs(policy, level="policy")
            cells.extend([f"{group.within.sum()}/{len(group)}", f"{group.gap_pp.abs().mean():.2f}"])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular*}\par\smallskip",
        r"\begin{minipage}{\columnwidth}\footnotesize The target approval share is 75\%. Within counts test years in 2012--2019 and 2021--2024 when the absolute Black--White approval gap was at most five percentage points. "
        r"Mean $|G|$ weights state-years equally. Pooled combines earlier cohorts. "
        r"Previous year uses the last available cohort's minimum adjustment. Each prior year requires one strength to meet tolerance in every earlier cohort.\end{minipage}\end{table}",
    ])
    return lines


def render(analysis):
    original = analysis == "original"
    source = "rolling_state_year.csv" if original else "from2010/rolling_state_year.csv"
    folds, years = load_folds(ROOT/"results/transfer"/source, 2014 if original else 2010)
    draw_figure(folds, years, ROOT/"paper/figures", "rolling_2014" if original else "fig5_rolling")
    if original:
        lines = original_window_table(folds, years)
    else:
        first = detailed_table(folds, years, STATES[:2], "tab:rollingfolds")
        second = detailed_table(folds, years, STATES[2:], "tab:rollingfolds_east")
        lines = first + [""] + second
    generated = ROOT/"paper/generated"
    generated.mkdir(parents=True, exist_ok=True)
    filename = "rolling_2014.tex" if original else "rolling_folds.tex"
    (generated/filename).write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not original:
        summary = summary_table(folds, ROOT/"results/transfer/from2010/rolling_previous.csv")
        (generated/"rolling_summary.tex").write_text("\n".join(summary)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=["primary", "original", "all"], default="all")
    requested = parser.parse_args().analysis
    for analysis in (["primary", "original"] if requested == "all" else [requested]):
        render(analysis)
