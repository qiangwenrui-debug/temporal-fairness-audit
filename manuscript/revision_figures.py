"""Render the four figures using the released, plot-level data.

From the manuscript-code directory, refresh the audit data and rerun data with
python revision_figures.py --refresh-from ../../input_results --policy-from ../../results/monitor
and render directly from the released figure CSVs with python revision_figures.py.
Run from any directory; relative command-line paths use the working directory.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SETTINGS = [
    ("credit", "Corporate credit", None, None),
    ("income_ca", "Income × race, CA", "results_income_race_CA_d1_quantile", "audit_inc_race_CA"),
    ("income_states", "Income × race, TX/NY/FL", "results_income_race_TX-NY-FL_d1_quantile", "audit_inc_race_TX-NY-FL"),
    ("employment_ca", "Employment × race, CA", "results_employment_race_CA_d1", "audit_emp_race_CA"),
    ("income_sex", "Income × sex, CA", "results_income_sex_CA_d1_quantile", "audit_inc_sex_CA"),
    ("employment_sex", "Employment × sex, CA", "results_employment_sex_CA_d1", "audit_emp_sex_CA"),
]
STATES = ["CA", "TX", "NY", "FL"]
COMMON_YEARS = [2017, 2018, 2019, 2021, 2022, 2023]
INK, MUTED, LINE = "#172c39", "#405762", "#c0cbd0"
BLUE, ORANGE, TEAL = "#14688c", "#ae491b", "#227a66"
BAND, GREY = "#e9eeef", "#69747a"
WIDTH = 7.16

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "font.weight": "regular", "axes.titlesize": 8.4, "axes.titleweight": "semibold",
    "axes.labelcolor": INK, "axes.edgecolor": "#86949a", "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    "legend.fontsize": 7.3, "figure.facecolor": "white", "axes.facecolor": "white",
})


def xgb_main(frame):
    return frame.loc[frame.model.eq("xgb") & frame.q.eq(.75)].copy()


def refresh(archive, target):
    from results_data import YEARLY

    target.mkdir(parents=True, exist_ok=True)
    audits, freeze = [], []
    checks = archive / "revision_results"
    credit = archive / "csmar_timing_check"
    yearly = pd.read_csv(credit / "timing_check_yearly.csv").query("design == 'base' and model == 'XGB'").sort_values("year")
    pooled = pd.read_csv(credit / "timing_check.csv").query("design == 'base' and model == 'XGB'").iloc[0]
    prior = YEARLY["csmar"]
    if yearly.year.tolist() != prior["years"] or pooled.beta_pool != 0:
        raise ValueError("Check the corporate year mapping and the frozen zero-calibration rule.")
    if not np.allclose(yearly.gap, prior["G"], atol=.00051):
        raise ValueError("The corporate interval snapshot refers to different estimates.")
    corp = yearly.rename(columns={"gap": "G", "gap_good": "RG"})
    corp = corp.assign(setting="credit", G_lo=prior["G_lo"], G_hi=prior["G_hi"],
                       RG_lo=np.nan, RG_hi=np.nan, pooled_G=pooled.pooled)
    audits.append(corp)
    freeze.append(corp.assign(beta_pool=pooled.beta_pool, G0=corp.G, G_bpool=corp.G))

    for key, _, folder, check_name in SETTINGS[1:]:
        source = archive / folder
        annual = xgb_main(pd.read_csv(source / "yearly.csv")).sort_values("year")
        pooled = xgb_main(pd.read_csv(source / "pooled.csv")).iloc[0]
        audits.append(annual.assign(setting=key, pooled_G=pooled.pooled_gap))
        evaluated = pd.read_csv(checks / check_name / "yearly.csv").query("model == 'xgb'").sort_values("year")
        if annual.year.tolist() != evaluated.year.tolist() or not np.allclose(annual.G, evaluated.G0):
            raise ValueError(f"Audit and applied-policy records disagree for {key}.")
        freeze.append(evaluated.assign(setting=key))

    for filename, parts, columns in [
        ("fig2_annual_audit.csv", audits, ["setting", "year", "G", "G_lo", "G_hi", "RG", "RG_lo", "RG_hi", "pooled_G"]),
        ("fig3_frozen_policy.csv", freeze, ["setting", "year", "beta_star", "beta_pool", "G0", "G_bpool"]),
    ]:
        out = pd.concat([part[columns] for part in parts], ignore_index=True)
        if out.duplicated([columns[0], "year"]).any():
            raise ValueError(f"Repeated setting-year in {filename}.")
        out.to_csv(target / filename, index=False, float_format="%.12g")


def save(fig, out, index):
    fig.savefig(out / f"fig{index}.pdf", pad_inches=.02, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"fig{index}.png", dpi=420, pad_inches=.02)
    plt.close(fig)


def figure_one(out):
    fig = plt.figure(figsize=(WIDTH, 2.95))
    ax = fig.add_axes([.012, .015, .976, .965])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    col = [.013, .359, .705]
    span = .282

    def card(x, y, height, title, lines, fill, edge):
        ax.add_patch(FancyBboxPatch(
            (x, y), span, height, boxstyle="round,pad=0.007,rounding_size=0.012",
            fc=fill, ec=edge, lw=.78,
        ))
        ax.text(x+.010, y+height-.034, title, va="top", size=8, weight="bold", color=edge)
        ax.text(x+.010, y+height-.104, lines, va="top", size=7.25, linespacing=1.27)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": "-|>", "color": MUTED, "lw": .85, "mutation_scale": 8,
            "shrinkA": 0, "shrinkB": 0,
        })

    for x, title in zip(col, ("Available information", "Decision in period t", "After labels arrive")):
        ax.text(x, .975, title, va="top", fontsize=9.1, weight="bold")
    for y, title, color in [(.84, "AUDIT", BLUE), (.405, "MONITOR", TEAL)]:
        ax.text(col[0], y, title, fontsize=7.4, weight="bold", color=color, va="bottom")
        ax.plot([.13, .987], [y+.022, y+.022], color=LINE, lw=.65)
    card(col[0], .505, .295, "Train the audit model",
         "Train on years through t − d.\nChoose β on pooled history or\nrequire each prior year to pass.", "#f3f7fa", BLUE)
    card(col[1], .505, .295, "Measure the approval gap",
         "Set group cutoffs from current\nscores at the chosen β. Check\nwhether |Gₜ(β)| ≤ δ.", "#f3f7fa", BLUE)
    card(col[2], .505, .295, "Decompose the annual gap",
         "Measure RGₜ among those with\nfavourable outcomes. The\nremainder is RCₜ = Gₜ − RGₜ.", "#f3f7fa", BLUE)
    card(col[0], .057, .31, "Reserve a reference cohort",
         "The last labelled year is H.\nTrain on earlier years. Use H\nas the reference cohort.", "#f2f7f4", TEAL)
    card(col[1], .057, .31, "Estimate the residual",
         "Recalibrate scores using H or\nweight its records to match\nthe current cohort.", "#f2f7f4", TEAL)
    card(col[2], .057, .31, "Check the estimate and alert",
         "Compare the estimated residual\nand its sensitivity interval with\nthe residual observed in year t.", "#f2f7f4", TEAL)
    for y in [.653, .213]:
        for start, end in zip(col, col[1:]):
            arrow((start+span+.012, y), (end-.012, y))
    save(fig, out, 1)


def annual_index(frame, start, end):
    return frame.set_index("year").reindex(range(start, end+1))


def year_ticks(ax, years):
    ax.set_xticks(years, [f"’{year % 100:02}" for year in years])


def figure_two(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.72), sharey=True)
    fig.subplots_adjust(left=.088, right=.988, bottom=.18, top=.93, wspace=.17, hspace=.46)
    for i, (ax, (key, label, _, _)) in enumerate(zip(axes.flat, SETTINGS)):
        rows = data[data.setting.eq(key)]
        annual = annual_index(rows, 2014 if i == 0 else 2015, 2022 if i == 0 else 2023)
        x = annual.index.to_numpy()
        ax.axhspan(-5, 5, color=BAND, lw=0)
        ax.axhline(0, color=LINE, lw=.64)
        ax.axhline(100 * rows.pooled_G.iloc[0], color=INK, lw=.95, ls=(0, (1, 1.7)))
        for name, color, marker, style in [("G", BLUE, "o", "-"), ("RG", ORANGE, "s", "--")]:
            lo, hi = annual[f"{name}_lo"], annual[f"{name}_hi"]
            if lo.notna().any():
                ax.fill_between(x, 100*lo, 100*hi, color=color, alpha=.16, lw=0)
            ax.plot(x, 100*annual[name], color=color, marker=marker, ms=3.2, lw=1.25,
                    ls=style, markeredgewidth=.55, markeredgecolor="white")
        ax.set_title(f"({chr(97+i)}) {label}", loc="left", fontsize=8.1)
        ax.set_ylim(-22, 12.3)
        ax.set_yticks([-20, -15, -10, -5, 0, 5, 10])
        year_ticks(ax, [2014, 2016, 2018, 2020, 2022] if i == 0 else [2015, 2017, 2019, 2021, 2023])
        if i % 3 == 0:
            ax.set_ylabel("Gap (pp)")
    handles = [
        Line2D([], [], color=BLUE, marker="o", label="Approval gap", ms=4),
        Line2D([], [], color=ORANGE, ls="--", marker="s", label="Residual gap", ms=4),
        Line2D([], [], color=INK, ls=":", label="Pooled approval gap"),
        Patch(facecolor=BAND, edgecolor=LINE, label="±5 pp approval tolerance"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, .026), columnspacing=1.2, handlelength=2.2)
    save(fig, out, 2)


def figure_three(data, out):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.95))
    fig.subplots_adjust(left=.015, right=.99, top=.91, bottom=.075, hspace=.48)
    years = list(range(2014, 2024))
    for panel, ax in enumerate(axes):
        ax.set_xlim(-3.45, 11.1)
        ax.set_ylim(5.55, -1.1)
        ax.axis("off")
        ax.set_title(["(a) Minimum calibration strength by year", "(b) Annual approval gap at the pooled calibration strength (pp)"][panel], loc="left", pad=2)
        for x, year in enumerate(years):
            ax.text(x, -.72, str(year), ha="center", va="center", size=7.3, weight="bold")
        ax.text(10.22, -.72, "Pooled β*" if panel == 0 else "Within ±5 pp", ha="center",
                va="center", size=7.2, weight="bold")
        for y, (key, label, _, _) in enumerate(SETTINGS):
            df = data.loc[data.setting.eq(key)].set_index("year")
            ax.text(-3.40, y, label, ha="left", va="center", size=7.2)
            for x, year in enumerate(years):
                valid = year in df.index
                ink, shade, weight = INK, "#ffffff", "normal"
                val = "–"
                if valid:
                    row = df.loc[year]
                    if panel == 0:
                        v = row.beta_star
                        shade = plt.cm.Blues(.06 + .27*v/.76) if v > 0 else "#ffffff"
                        val = f"{v:.3f}"
                    else:
                        v = row.G_bpool * 100
                        outside = abs(v) > 5+1e-9
                        shade, ink, weight = ("#f6d9c8", "#683316", "bold") if outside else ("#e6f1ed", "#224b43", "normal")
                        val = f"{v:+.1f}".replace("-", "−")
                else:
                    ink = "#8c9a9e"
                ax.add_patch(Rectangle((x-.48, y-.41), .96, .82, facecolor=shade,
                                       edgecolor="none"))
                ax.text(x, y, val, ha="center", va="center", size=7.45, color=ink, weight=weight)
            if panel == 0:
                stat = f"{df.beta_pool.iloc[0]:.3f}"
            else:
                before = (df.G0.abs() <= .05+1e-11).sum()
                after = (df.G_bpool.abs() <= .05+1e-11).sum()
                stat = f"{before}/{len(df)} → {after}/{len(df)}"
            ax.text(10.24, y, stat, ha="center", va="center", size=7.2)
        ax.axvline(9.57, ymin=.03, ymax=.99, color=LINE, lw=.6)
    fig.text(.018, .013, "Dash = no test cohort. Orange cells exceed ±5 pp. The last column of (b) counts years inside the band at β = 0 and at the pooled strength.", size=7.0, color=MUTED)
    save(fig, out, 3)


def collect_policy(root, target):
    """Read one completed seed-0 launch per state; check the co-timed decision and truth."""
    parts = []
    for state in STATES:
        folder = root / f"{state}_income_race_quantile_d1_s0"
        keys = ["state", "year", "policy"]
        values = pd.read_csv(folder/"yearly.csv")
        values = values[values.policy.isin(["none", "per_period"])].copy()
        intervals = pd.read_csv(folder/"intervals.csv").query("gamma_method == 'history'")
        intervals = intervals[intervals.policy.isin(["none", "per_period"])]
        frame = values.merge(intervals[keys+["true_RG_pp", "lo_pp", "hi_pp", "action", "actual", "incorrect_definite"]],
                             on=keys, how="left", validate="one_to_one")
        checked = frame.dropna(subset=["action"])
        if not np.allclose(checked.true_RG*100, checked.true_RG_pp):
            raise ValueError(f"Mismatched decision rule for {state}.")
        original = set(frame[frame.year.le(2023)].year)
        if original != {2016, 2017, 2018, 2019, 2021, 2022, 2023}:
            raise ValueError(f"Incomplete original window for {state}.")
        if not (frame.groupby("year").policy.nunique() == 2).all():
            raise ValueError(f"Unpaired rules for {state}.")
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    n24 = combined.loc[combined.year.eq(2024), "state"].nunique()
    if n24 not in (0,4):
        raise ValueError("Finish or separate the 2024 extension before plotting.")
    fields = ["state", "year", "holdout", "policy", "beta", "true_G", "true_RG",
              "E2", "E3", "lo_pp", "hi_pp", "action", "actual", "incorrect_definite"]
    target.mkdir(parents=True, exist_ok=True)
    combined[fields].to_csv(target/"fig4_policy_check.csv", index=False, float_format="%.12g")
    secondary = pd.read_csv(root / "q060" / "all_yearly.csv")
    secondary = secondary.loc[
        secondary.state.isin(STATES) & secondary.year.isin(COMMON_YEARS+[2024]) &
        secondary.task.eq("income") & secondary.group.eq("race") & secondary.label.eq("quantile") &
        secondary.delay.eq(1) & secondary.seed.eq(0) & secondary.q.eq(.60) &
        secondary.policy.eq("per_period"),
        ["state", "year", "q", "policy", "beta", "true_RG", "E3"],
    ].sort_values(["state", "year"])
    expected = pd.MultiIndex.from_product([STATES, COMMON_YEARS+[2024]], names=["state", "year"])
    records = secondary.set_index(["state","year"])
    if not records.index.is_unique or records.reindex(expected).isna().any().any():
        raise ValueError("Incomplete q=.60 calibrated state-year decisions.")
    secondary.to_csv(target/"fig4_q060_alerts.csv", index=False, float_format="%.12g")


def policy_figure(data, secondary, out):
    """End-to-end calibration and monitoring from a single, common set of reruns."""
    fig = plt.figure(figsize=(WIDTH, 4.75))
    grid = fig.add_gridspec(2, 2, left=.083, right=.986, top=.935, bottom=.155,
                            wspace=.23, hspace=.92, height_ratios=[1.45, 1])
    finish = int(data.year.max())
    years = COMMON_YEARS
    display = years + ([2024] if finish == 2024 else [])
    lines = [("none","true_RG", "Observed, unadjusted rule", GREY, ":", "v"),
             ("per_period","true_RG", "Observed, calibrated rule", INK, "-", "o"),
             ("per_period","E2", "Recalibrated scores", ORANGE, "--", "s"),
             ("per_period","E3", "Weighted reference", BLUE, "-.", "D")]
    selected = data.loc[data.state.isin(["CA", "FL"])]
    calibrated = selected.loc[selected.policy.eq("per_period")]
    bounds = calibrated[["lo_pp","hi_pp"]].dropna().to_numpy().ravel()
    extremes = np.r_[selected.true_RG.to_numpy()*100, calibrated[["E2","E3"]].to_numpy().ravel()*100, bounds, -2, 2]
    lower, upper = np.floor(np.min(extremes) - .35), np.ceil(np.max(extremes) + .35)
    for i, (state, label) in enumerate([("CA", "California"), ("FL", "Florida")]):
        ax = fig.add_subplot(grid[0,i])
        subset = data[data.state.eq(state)]
        intervals = subset[subset.policy.eq("per_period")].dropna(subset=["lo_pp","hi_pp"])
        whisker = "#677783"
        ax.vlines(intervals.year, intervals.lo_pp, intervals.hi_pp, color=whisker, lw=.86, zorder=2)
        for bound in ["lo_pp","hi_pp"]:
            ax.hlines(intervals[bound], intervals.year-.115, intervals.year+.115, color=whisker, lw=.86, zorder=2)
        for rule, key, _, color, ls, marker in lines:
            annual = annual_index(subset[subset.policy.eq(rule)], 2016, finish)
            ax.plot(annual.index, annual[key]*100, color=color, ls=ls, marker=marker, ms=3.1,
                    lw=1.6 if color == INK else 1.07, zorder=4 if color == INK else 3)
        for cut in (-2, 2):
            ax.axhline(cut, color=TEAL, lw=.70, ls=(0, (3.5,2)))
        if finish == 2024:
            ax.axvline(2023.5, color=LINE, lw=.7)
        ax.set_ylim(lower, upper)
        ax.set_yticks(np.arange(np.ceil(lower/2)*2, upper+.01, 2))
        year_ticks(ax, [2016]+display)
        ax.set_title(f"({chr(97+i)}) {label}, q = .75", loc="left")
        if i == 0:
            ax.set_ylabel("Residual gap (pp)")
        else:
            ax.tick_params(labelleft=False)
    legend = [Line2D([], [], color=c, ls=ls, marker=mark, ms=3.3, lw=1.3, label=lab)
              for _, _, lab, c, ls, mark in lines]
    whisker_handle = ax.errorbar([], [], yerr=[], color=whisker, fmt="none", lw=.86, capsize=2.2, label="Historical sensitivity interval")
    legend += [whisker_handle, Line2D([], [], color=TEAL, ls="--", lw=.8, label="±2 pp alert threshold")]
    fig.legend(handles=legend, ncol=3, loc="upper center", bbox_to_anchor=(.51,.558),
               frameon=False, fontsize=7.05, columnspacing=1.0, handlelength=1.65)

    ax = fig.add_subplot(grid[1,0])
    paired = data[data.year.isin(years)].copy()
    for m in ["E2","E3"]:
        paired[m] = 100*(paired[m]-paired.true_RG).abs()
    mae = paired.groupby(["state","policy"])[["E2","E3"]].mean()
    mae_columns = [("none","E2"), ("none","E3"), ("per_period","E2"), ("per_period","E3")]
    ax.set_xlim(-.5,3.5)
    ax.set_ylim(3.5,-.5)
    ax.set_title("(c) Mean absolute error (pp), q = .75", loc="left", pad=5)
    ax.set_xticks(range(4), ["Recalibrated\nscores", "Weighted\nreference"] * 2)
    for x, label in ((.25, "Unadjusted rule"), (.75, "Annual calibration")):
        ax.text(x, -.39, label, transform=ax.transAxes, ha="center", va="top", fontsize=7.4)
    ax.axvline(1.5, color=GREY, lw=.7)
    ax.set_yticks(range(4), STATES)
    vmax = mae.to_numpy().max()
    for r, state in enumerate(STATES):
        for j, (rule, method) in enumerate(mae_columns):
            val = mae.loc[(state,rule),method]
            color = plt.cm.Blues(.075 + .31*val/vmax)
            ax.add_patch(Rectangle((j-.49,r-.48),.98,.96,fc=color,ec="none"))
            ax.text(j,r,f"{val:.2f}",ha="center",va="center",fontsize=7.6)
    ax.tick_params(length=0, pad=3.8)
    ax.spines[:].set_visible(False)
    ax = fig.add_subplot(grid[1,1])
    ax.set_title("(d) Weighted-reference alerts, q = .60", loc="left", pad=5)
    ax.set_xlim(-.5,len(display)-.5)
    ax.set_ylim(3.5,-.5)
    ax.set_xticks(range(len(display)), [f"’{y%100:02}" for y in display])
    ax.set_yticks(range(4), STATES)
    ax.tick_params(length=0, pad=3.8)
    ax.spines[:].set_visible(False)
    alerts = secondary.set_index(["state","year"])
    classes = {(True,True):("Hit","#b6dbd1"), (True,False):("Miss","#efb5a1"),
               (False,True):("False\nalert","#f4deb2"), (False,False):("No alert","#ebeff0")}
    for r, state in enumerate(STATES):
        for j, year in enumerate(display):
            row = alerts.loc[(state,year)]
            label, shade = classes[(abs(row.true_RG)>.02, abs(row.E3)>.02)]
            ax.add_patch(Rectangle((j-.49,r-.48),.98,.96,fc=shade,ec="none"))
            ax.text(j,r,label,ha="center",va="center",size=6.15,
                    weight="bold" if label != "No alert" else "normal",color=INK)
    if finish == 2024:
        ax.axvline(len(display)-1.5, color=GREY, lw=.7)
    fig.text(.083,.017,"In (d), hits and misses refer to observed gaps outside ±2 pp. Vertical lines separate the 2024 test cohorts.",size=7.15,color=MUTED)
    save(fig, out, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-from", type=Path)
    parser.add_argument("--policy-from", type=Path, help="Root containing the four completed policy-conditional monitor runs.")
    parser.add_argument("--data", type=Path, default=HERE/"figure_data")
    parser.add_argument("--out", type=Path, default=HERE.parents[1]/"paper"/"figures")
    args = parser.parse_args()
    if args.refresh_from:
        refresh(args.refresh_from, args.data)
    if args.policy_from:
        collect_policy(args.policy_from, args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    figure_one(args.out)
    figure_two(pd.read_csv(args.data/"fig2_annual_audit.csv"), args.out)
    figure_three(pd.read_csv(args.data/"fig3_frozen_policy.csv"), args.out)
    policy_figure(pd.read_csv(args.data/"fig4_policy_check.csv"),
                  pd.read_csv(args.data/"fig4_q060_alerts.csv"), args.out)


if __name__ == "__main__":
    main()
