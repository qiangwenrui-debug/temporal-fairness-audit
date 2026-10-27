"""Figures 1-4 for the manuscript (300 dpi PNG + PDF). Data: results_data.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
from results_data import YEARLY, LABEL_ROBUST, MON_YEARS, MON_CA

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
serif = [f for f in ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
         if any(f in x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams.update({
    "font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, BAND = "#222222", "#6b6b6b", "#ececec"
ORDER = ["csmar", "inc_race_ca", "inc_race_tx", "emp_race_ca", "inc_sex_ca"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def year_axis(ax, years):
    ax.set_xticks(years)
    ax.set_xticklabels([f"'{str(y)[2:]}" for y in years])


# ---------------------------------------------------------------- Fig. 1 pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(7.16, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f4f7fc", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, weight="bold",
                color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.6, color=INK,
                linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED, mutation_scale=8))

    ax.text(1, 29.5, "Decision time (period t)", fontsize=7.5, color=BLUE, weight="bold", va="top")
    ax.text(1, 9.5, "After labels\narrive\n(period t + d)", fontsize=7.5, color=ORANGE,
            weight="bold", va="top", linespacing=1.2)
    box(1, 15.5, 18, 11, "1. Retrain", "model $m_t$ on\nperiods $\\leq t-d$")
    box(23, 15.5, 18, 11, "2. Score & screen", "approve top $q$;\nthresholds $\\tau_{gt}(\\beta)$")
    box(45, 15.5, 22, 11, "3. Approval gap", "$G_t(\\beta)$ and $\\beta^*_t$\n(Algorithm 1; no labels)")
    box(71, 15.5, 28, 11, "4. Residual estimate (Sec. IV)",
        "$\\widehat{RG}_t$ from latest labelled\nperiod $H$, reweighted (E3)")
    arrow(19.6, 21, 22.4, 21); arrow(41.6, 21, 44.4, 21); arrow(67.6, 21, 70.4, 21)
    box(23, 0.5, 30, 10.5, "5. Decompose (Prop. 2)",
        "$G_t = RG_t + RC_t$;\nbootstrap CIs", fc="#fdf3ee", ec=ORANGE)
    box(57, 0.5, 42, 10.5, "6. Aggregate & compare (Prop. 1)",
        "pooled gap vs. period gaps; $\\beta^*_{pool}$ vs. $\\beta^*_t$;\n"
        "$\\widehat{RG}_t$ vs. realized $RG_t$", fc="#fdf3ee", ec=ORANGE)
    arrow(32, 15.2, 38, 11.3); arrow(85, 15.2, 85, 11.3); arrow(53.6, 5.8, 56.4, 5.8)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Fig. 2 yearly gaps
def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1))
    for i, (ax, key) in enumerate(zip(axes.flat[:5], ORDER)):
        d = YEARLY[key]; yrs = np.array(d["years"])
        G, lo, hi = 100 * np.array(d["G"]), 100 * np.array(d["G_lo"]), 100 * np.array(d["G_hi"])
        ax.axhspan(-5, 5, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
        ax.axhline(100 * d["pooled_G"], color=INK, lw=0.8, ls=":", zorder=2)
        ax.fill_between(yrs, lo, hi, color=BLUE, alpha=0.18, lw=0, zorder=2)
        ax.plot(yrs, G, color=BLUE, lw=1.4, marker="o", ms=3, zorder=3)
        ax.plot(yrs, 100 * np.array(d["RG"]), color=ORANGE, lw=1.2, ls="--", marker="s", ms=2.6,
                zorder=3)
        ax.set_title(f"({'abcde'[i]}) {d['label']}", loc="left", fontsize=7.5)
        year_axis(ax, yrs)
        if i % 3 == 0:
            ax.set_ylabel("Gap (pp)")
    # robustness panel
    ax = axes.flat[5]
    ax.axhspan(-5, 5, color=BAND, zorder=0, lw=0); ax.axhline(0, color=MUTED, lw=0.5)
    styles = [(BLUE, "-", "o"), (ORANGE, "--", "s"), (AQUA, "-.", "^"), (INK, ":", "D")]
    for (name, v), (c, ls, mk) in zip(LABEL_ROBUST.items(), styles):
        ax.plot(v["years"], 100 * np.array(v["G"]), color=c, ls=ls, marker=mk, ms=2.6, lw=1.1,
                label=name.replace("$", r"\$"))
    ax.set_title("(f) CA income × race: label and delay", loc="left", fontsize=7.5)
    year_axis(ax, ACS := YEARLY["inc_race_ca"]["years"])
    ax.legend(loc="lower left", frameon=False, fontsize=6, handlelength=2.2)
    # shared legend for panels a-e
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.4, label="Approval gap $G_t$ (95% CI shaded)"),
               Line2D([], [], color=ORANGE, marker="s", ms=2.6, lw=1.2, ls="--",
                      label="Residual among qualified $RG_t$"),
               Line2D([], [], color=INK, lw=0.8, ls=":", label="Pooled gap"),
               Patch(color=BAND, label="±5 pp tolerance")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=0.8)
    save(fig, "fig2_yearly_gaps")


# ---------------------------------------------------------------- Fig. 3 beta*
def fig3():
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.85), sharey=True)
    for i, (ax, key) in enumerate(zip(axes, ORDER)):
        d = YEARLY[key]; yrs = d["years"]
        ax.axhline(d["pooled_beta"], color=ORANGE, lw=1.2, ls="--", zorder=2)
        ax.vlines(yrs, 0, d["beta"], color=BLUE, lw=0.8, alpha=0.5, zorder=1)
        ax.plot(yrs, d["beta"], ls="none", marker="o", ms=3.4, color=BLUE, zorder=3)
        ax.set_title(f"({'abcde'[i]}) {d['label']}", loc="left", fontsize=6.8)
        ax.set_ylim(-0.03, 0.85)
        ax.set_xticks(yrs[::2]); ax.set_xticklabels([f"'{str(y)[2:]}" for y in yrs[::2]])
    axes[0].set_ylabel(r"Calibration strength $\beta$")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="none", ms=3.4,
                               label=r"Required in period $t$: $\beta^*_t$"),
                        Line2D([], [], color=ORANGE, ls="--", lw=1.2,
                               label=r"Set once on pooled data: $\beta^*_{pool}$")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    save(fig, "fig3_beta_mismatch")


# ---------------------------------------------------------------- Fig. 4 monitoring
def fig4():
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    yrs = MON_YEARS
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axhline(-2, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.text(2016, -1.93, "alert threshold", fontsize=6, color=MUTED, va="bottom")
    ax.plot(yrs, 100 * np.array(MON_CA["true"]), color=INK, lw=2, marker="o", ms=4,
            label="Realized $RG_t$ (after labels)", zorder=5)
    spec = [("E0", "E0 last observed", MUTED, ":", "v"),
            ("E1", "E1 soft labels", YELLOW, "--", "^"),
            ("E2", "E2 recalibrated", AQUA, "-.", "s"),
            ("E3", "E3 covariate-weighted", BLUE, "-", "D")]
    for k, lab, c, ls, mk in spec:
        ax.plot(yrs, 100 * np.array(MON_CA[k]), color=c, ls=ls, marker=mk, ms=3, lw=1.1, label=lab)
    ax.set_xticks(yrs); ax.set_xticklabels([f"'{str(y)[2:]}" for y in yrs])
    ax.set_ylabel("Residual gap (pp)"); ax.set_xlabel("Test year")
    ax.legend(loc="lower left", frameon=False, fontsize=6, handlelength=2.4)
    fig.tight_layout()
    save(fig, "fig4_monitoring")


if __name__ == "__main__":
    print("serif fonts:", serif[:2])
    fig1(); fig2(); fig3(); fig4()
    import json
    from PIL import Image
    json.dump({f: Image.open(f"{OUT}/{f}").size for f in sorted(os.listdir(OUT)) if f.endswith(".png")},
              open("figsizes.json", "w"))
    print(sorted(os.listdir(OUT)))
