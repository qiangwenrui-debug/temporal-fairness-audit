"""Render the four-state comparison of fixed and historically selected margins."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from revision_figures import BLUE, INK, LINE, MUTED, STATES, WIDTH
from rolling_outputs import STATE_NAMES, gap_cell


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fixed_0", "fixed_0_5", "fixed_1", "rolling")
LABELS = ("0 pp", "0.5 pp", "1 pp", "Selected")
TEST_YEARS = [*range(2012, 2020), *range(2021, 2025)]


def read_folds():
    source = ROOT / "results/transfer/from2010/rolling_margin_yearly.csv"
    frame = pd.read_csv(source)
    folds = frame.set_index(["state", "year", "policy"]).sort_index()
    grid = pd.MultiIndex.from_product([STATES, TEST_YEARS, METHODS], names=folds.index.names)
    if not folds.index.is_unique or len(folds) != len(grid) or len(grid.difference(folds.index)):
        raise ValueError("Expected the four rules for every state and test year.")
    if not folds.within.eq(folds.gap_pp.abs() <= 5).all():
        raise ValueError("Test classifications must use the unrounded five-point tolerance.")
    if not folds.changed_vs_common.eq(folds.added_vs_common + folds.removed_vs_common).all():
        raise ValueError("Changed decisions must include new approvals and new rejections.")
    fixed = folds.xs("fixed_0", level="policy")
    if not fixed.changed_vs_common.eq(0).all():
        raise ValueError("The zero-margin rule must agree with the comparison decisions.")
    for method in METHODS:
        block = folds.xs(method, level="policy")
        np.testing.assert_array_equal(block.baseline_within, fixed.within)
        np.testing.assert_array_equal(block.n, fixed.n)
        np.testing.assert_array_equal(block.newly_within, block.within & ~fixed.within)
        np.testing.assert_array_equal(block.newly_outside, ~block.within & fixed.within)
    return folds


def state_totals(folds):
    totals = folds.groupby(level=["state", "policy"]).agg(
        rescued=("newly_within", "sum"), harmed=("newly_outside", "sum"),
        changed=("changed_vs_common", "sum")
    )
    totals["avoided"] = totals.rescued - totals.harmed
    return totals


def matrix_axes(ax, ncol, labels):
    ax.set_xlim(-.5, ncol-.5)
    ax.set_ylim(len(STATES)-.5, -.5)
    ax.set_yticks(range(len(STATES)), STATES)
    ax.set_xticks(range(ncol), labels)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=3, labelsize=7.25)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def summary_panel(ax, entries, metric, title):
    matrix_axes(ax, len(METHODS), LABELS)
    values = entries[metric]
    scale = max(1, values.abs().max())
    colours = LinearSegmentedColormap.from_list("counts", ["#f0f4f5", "#a9cad6", BLUE])
    for i, state in enumerate(STATES):
        for j, method in enumerate(METHODS):
            value = int(values.loc[state, method])
            shade = abs(value)/scale
            ax.add_patch(Rectangle((j-.485, i-.474), .97, .948,
                                   facecolor=colours(shade), edgecolor="none"))
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if shade > .84 else INK,
                    fontsize=8.25, fontweight="bold" if method=="rolling" else "normal")
    ax.set_title(title, loc="left", fontsize=8.9, pad=18)


def figure(folds):
    totals = state_totals(folds)
    fig = plt.figure(figsize=(WIDTH, 3.25))
    grid = fig.add_gridspec(2, 2, left=.067, right=.995, top=.848, bottom=.091,
                            wspace=.23, hspace=.83, height_ratios=[1.15,1])
    summary_panel(fig.add_subplot(grid[0,0]), totals, "avoided", "(a) Exceedances avoided vs. zero margin")
    summary_panel(fig.add_subplot(grid[0,1]), totals, "changed", "(b) Decisions changed vs. zero margin")
    ax = fig.add_subplot(grid[1,:])
    years = range(TEST_YEARS[0],TEST_YEARS[-1]+1)
    matrix_axes(ax, len(years), [str(year) for year in years])
    for i, state in enumerate(STATES):
        for j, year in enumerate(years):
            if year not in TEST_YEARS:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=7.6)
                continue
            row = folds.loc[state, year, "rolling"]
            edge = LINE if row.within else INK
            face = "white" if row.within else "#e0eaf0"
            ax.add_patch(Rectangle((j-.474, i-.454), .948, .908, facecolor=face,
                                   edgecolor=edge, linewidth=.45 if row.within else 1.35))
            ax.text(j, i, f"{row.margin_pp:g}", va="center", ha="center", fontsize=8,
                    color=INK, fontweight="normal" if row.within else "bold")
    ax.set_title("(c) Margin selected before each test year (pp)", loc="left", fontsize=8.9, pad=18)
    fig.text(.067,.015,"Heavy borders mark gaps outside ±5 pp. No standard ACS sample was available for 2020.",
             fontsize=7.1,color=MUTED)
    out = ROOT / "paper/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"rolling_margin.pdf", metadata={"CreationDate":None, "ModDate":None})
    fig.savefig(out/"rolling_margin.png", dpi=360)
    plt.close(fig)


def fold_table(folds, states, label):
    state_names = " and ".join(STATE_NAMES[s] for s in states)
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Historical Margins and Test-Year Approval Gaps in the Income Screen in " + state_names + r"}\label{" + label + "}",
        r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular*}{.99\textwidth}{@{\extracolsep{\fill}}llrr rrrr rrr@{}}\toprule",
        r" & & & Selected & \multicolumn{4}{c}{Approval gap (pp)}"
        r" & \multicolumn{3}{c}{Decisions changed from zero margin} \\",
        r"\cmidrule(lr){5-8}\cmidrule(l){9-11}",
        r"State & Year & People & margin (pp) & 0 pp & 0.5 pp & 1 pp & Selected"
        r" & 0.5 pp & 1 pp & Selected \\\midrule",
    ]
    for index,state in enumerate(states):
        if index:
            lines.append(r"\midrule")
        for year in TEST_YEARS:
            group = folds.loc[(state,year)]
            chosen = group.loc["rolling"]
            cells = [state if year==TEST_YEARS[0] else "", str(year), f"{chosen.n:,.0f}", f"{chosen.margin_pp:g}"]
            cells.extend(gap_cell(group.loc[method]) for method in METHODS)
            cells.extend(f"{group.loc[method].changed_vs_common:,.0f}" for method in METHODS[1:])
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular*}\par\smallskip",
        r"\begin{minipage}{.99\textwidth}\footnotesize Margins of 0, 0.5 and 1 pp"
        r" give historical tolerances of 5, 4.5 and 4 pp, respectively. The selected margin uses earlier test years in the same state. All four rules"
        r" use the same test-year scores and a 75\% target approval share. The gap is the Black approval rate minus the White approval rate,"
        r" in percentage points. Bold entries exceed five points in absolute value before rounding. Each change count"
        r" compares decisions for the same people under the indicated rule and the zero-margin rule,"
        r" counting both new approvals and new rejections.\end{minipage}\end{table*}",
    ])
    return lines


def main():
    folds = read_folds()
    figure(folds)
    tables = fold_table(folds, STATES[:2], "tab:margin_west")
    tables += [""] + fold_table(folds, STATES[2:], "tab:margin_east")
    generated = ROOT / "paper/generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated/"rolling_margin_folds.tex").write_text("\n".join(tables)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
