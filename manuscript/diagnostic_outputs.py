"""Draw matched score and threshold diagnostics for the four-state rolling income audit."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from revision_figures import BAND, BLUE, INK, MUTED, ORANGE, STATES, TEAL, WIDTH

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT/"results/transfer/from2010"
DIAG = TRANSFER/"diagnostics"
COMP = ROOT/"results/diagnostics"
FIG = ROOT/"paper/figures"
GROUP_COLORS = {"Black": BLUE, "White": ORANGE}
STATE_NAMES = {"CA": "California", "TX": "Texas", "NY": "New York", "FL": "Florida"}
YEARS = range(2012, 2025)
YEARTICKS = np.array([2012,2014,2016,2018,2020,2022,2024])


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def dates(ax, every_year=False):
    ax.set_xlim(2011.5, 2024.5)
    ticks = list(YEARS) if every_year else YEARTICKS
    ax.set_xticks(ticks, [f"’{y%100:02}" for y in ticks])


def export(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/(name+".pdf"),metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(FIG/(name+".png"),dpi=360)
    plt.close(fig)


def diagnostic_figure(cdfs, cuts, replay, compositions):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH,6.00), gridspec_kw={"height_ratios":[1,1,1.2]})
    fig.subplots_adjust(left=.084,right=.982,bottom=.116,top=.968,hspace=.73,wspace=.27)
    for index,(state,ax) in enumerate(zip(STATES,axes.flat[:4])):
        frame = cdfs[(cdfs.state==state) & (cdfs.year==2021)]
        common = cuts[(cuts.state==state) & (cuts.year==2021) & (cuts.policy=="common")].set_index("group")
        for group,color in GROUP_COLORS.items():
            series = frame[(frame.group==group)&frame.score.between(.019,.221)]
            ax.plot(series.score,series.pct_below,lw=1.25,color=color,
                    ls="-" if group=="Black" else "--",label=group)
            row = common.loc[group]
            ax.vlines(row.threshold, 5, row.rejection_pct, color=color, lw=.7,ls=(0,(2,2)))
            ax.plot(row.threshold,row.rejection_pct,"o" if group=="Black" else "s",
                    color=color,ms=4.2,mec="white",mew=.45)
        ax.set_xlim(.02,.22)
        ax.set_ylim(5,53)
        ax.set_xticks([.05,.10,.15,.20])
        ax.set_yticks([10,20,30,40,50])
        ax.set_title(f"({chr(97+index)}) {STATE_NAMES[state]}, 2021",loc="left",pad=3)
        if index in (0,2):
            ax.set_ylabel("Scores below cutoff (%)")
        if index in (2,3):
            ax.set_xlabel("Score cutoff")
        if index==0:
            ax.legend(loc="upper left",borderaxespad=.25,frameon=False,ncol=2,columnspacing=.8,handlelength=1.1)
    ax=axes[2,0]
    ny=cuts[(cuts.state=="NY") & (cuts.policy=="common") & (cuts.group=="Black")].set_index("year").reindex(YEARS)
    dates(ax)
    ax.plot(ny.index,ny.beta,color=INK,drawstyle="default",marker="s",ms=3.4,lw=.95,
            markerfacecolor="white",label="Minimum for every prior year")
    ax.plot(ny.index,ny.target_year_min_beta,color=ORANGE,marker="o",ms=3.2,lw=1.1,
            label="Minimum for the test year")
    for row in ny.dropna().itertuples():
        if abs(row.gap_pp)>5:
            ax.vlines(row.Index,row.beta,row.target_year_min_beta,color=ORANGE,lw=.8)
    ax.set_ylabel("Adjustment strength")
    ax.set_ylim(.485,.70)
    ax.set_yticks([.50,.55,.60,.65,.70])
    ax.set_title("(e) New York threshold adjustment",loc="left",pad=3)
    ax.set_xlabel("Test year")
    ax.legend(loc="upper left",frameon=False,handlelength=1.1,borderaxespad=.1,fontsize=6.6)
    ax=axes[2,1]
    z=replay.query("year==2021").set_index("state").loc[STATES]
    joint=compositions.query("year==2021 and scenario=='prior_joint'").set_index("state").loc[STATES]
    pos=np.arange(4)
    ax.axhline(-5,ls=(0,(3,2)),color=MUTED,lw=.8)
    series=[(z.prior_gap_pp,"2019 sample, 2019 model",MUTED,"o"),
            (z.old_model_target_gap_pp,"2021 sample, 2019 model",BLUE,"s"),
            (z.target_gap_pp,"2021 sample, 2021 model",ORANGE,"D"),
            (joint.gap_pp,"2021 sample, 2019 composition",TEAL,"^")]
    for offset,(points,label,color,mark) in zip([-.21,-.07,.07,.21],series):
        ax.plot(pos+offset,points.to_numpy(),ls="none",marker=mark,color=color,ms=4.2,label=label)
    ax.set_xticks(pos,STATES)
    ax.set_ylim(-11.5,-3.5)
    ax.set_yticks([-11,-9,-7,-5])
    ax.set_ylabel("Approval gap (pp)")
    ax.set_xlim(-.48,3.48)
    ax.set_title("(f) Holding the 2021 strength fixed",loc="left",pad=3)
    ax.legend(loc="upper center",bbox_to_anchor=(.45,-.19),ncol=2,frameon=False,
              columnspacing=.6,handlelength=.8,handletextpad=.28,fontsize=6.2)
    export(fig,"score_diagnostics")


def profile_figure(profiles,cuts,cohorts):
    fig, axes=plt.subplots(4,3,figsize=(WIDTH,6.10),gridspec_kw={"width_ratios":[1.02,1,1]})
    fig.subplots_adjust(left=.079,right=.987,bottom=.13,top=.943,hspace=.37,wspace=.24)
    for i,state in enumerate(STATES):
        subset=cohorts[cohorts.state==state]
        ax=axes[i,0]
        ax.set_ylim(0,66)
        ax.set_yticks([0,20,40,60])
        for group,color in GROUP_COLORS.items():
            row=subset[subset.group==group].set_index("year").reindex(YEARS)
            ax.plot(row.index,100*row.bachelor_share,color=color,lw=1.05,
                    ls="-" if group=="Black" else "--")
            if group=="Black":
                ax.plot(row.index,100*row.group_share,color=INK,ls=(0,(2,2)),lw=1.0)
        ax.set_ylabel(f"{state} · percent")
        for j,group in enumerate(GROUP_COLORS,1):
            ax=axes[i,j]
            g=profiles[(profiles.state==state)&(profiles.group==group)].set_index("year").reindex(YEARS)
            c=cuts[(cuts.state==state)&(cuts.group==group)&(cuts.policy=="common")].set_index("year").reindex(YEARS)
            ax.vlines(g.index,g.p10,g.p90,color="#adbcc2",lw=.7)
            ax.vlines(g.index,g.p25,g.p75,color="#70909b",lw=3.4)
            ax.scatter(g.index,g.p50,marker="_",color=INK,s=28,lw=1.1)
            ax.scatter(c.index,c.threshold,marker="D",color="#a43d11",s=9,zorder=3)
            ax.set_ylim(-.025,1)
            ax.set_yticks([0,.5,1])
            if j==2:
                ax.tick_params(labelleft=False)
        for ax in axes[i]:
            dates(ax)
            if i<3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Test year")
    for j,title in enumerate(["Sample composition","Black score distribution","White score distribution"]):
        axes[0,j].set_title(title,loc="left",pad=4)
    labels=[Line2D([],[],color=INK,ls=(0,(2,2)),label="Black share of cohort"),
            Line2D([],[],color=BLUE,label="Bachelor’s or higher, Black"),
            Line2D([],[],color=ORANGE,ls="--",label="Bachelor’s or higher, White"),
            Line2D([],[],color="#adbcc2",lw=1,label="10th–90th percentile"),
            Line2D([],[],color="#70909b",lw=3,label="25th–75th percentile"),
            Line2D([],[],color=INK,marker="_",ls="none",label="Median"),
            Line2D([],[],color="#a43d11",marker="D",ms=3,ls="none",label="Group approval cutoff")]
    fig.legend(handles=labels,loc="lower center",bbox_to_anchor=(.5,.007),ncol=4,frameon=False,
               handlelength=1.35,columnspacing=1,fontsize=6.5)
    export(fig,"score_profiles")


def contrast_figure(replay,compositions):
    model_ci=read(TRANSFER/"model_cohort_intervals.csv").query("policy=='common' and metric=='model_change_pp'")
    composition_ci=read(COMP/"composition_intervals.csv").query("policy=='common' and scenario=='prior_joint'")
    fig, axes=plt.subplots(2,2,figsize=(WIDTH,3.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.083,right=.98,bottom=.175,top=.93,wspace=.18,hspace=.32)
    joint=compositions.query("scenario=='prior_joint'").merge(composition_ci,on=["state","year"],validate="one_to_one")
    replay=replay.merge(model_ci[["state","year","estimate","lower_pp","upper_pp"]],
                        on=["state","year"],validate="one_to_one")
    if len(joint)!=48 or len(replay)!=48:
        raise ValueError("Both paired intervals must cover every state-year.")
    np.testing.assert_allclose(replay.model_change_pp,replay.estimate,atol=1e-12,rtol=0)
    for i,(ax,state) in enumerate(zip(axes.flat,STATES)):
        r=replay[replay.state==state].set_index("year").reindex(YEARS)
        c=joint[joint.state==state].set_index("year").reindex(YEARS)
        ax.axhline(0,color="#9caeb6",lw=.7)
        series=[(-.12,-r.model_change_pp,-r.upper_pp,-r.lower_pp,BLUE,"s","Use preceding model"),
                (.12,c.delta_from_observed_pp,c.delta_lower_pp,c.delta_upper_pp,ORANGE,"o","Restore preceding composition")]
        for offset,points,lo,hi,color,marker,label in series:
            x=points.index+offset
            ax.vlines(x,lo,hi,color=color,lw=.7,alpha=.7)
            ax.hlines(lo,x-.065,x+.065,color=color,lw=.7,alpha=.7)
            ax.hlines(hi,x-.065,x+.065,color=color,lw=.7,alpha=.7)
            ax.plot(x,points,color=color,lw=.8,marker=marker,ms=3.2,
                    mfc="white" if marker=="s" else color,label=label)
        dates(ax)
        ax.set_title(f"({chr(97+i)}) {STATE_NAMES[state]}",loc="left")
        if i in (0,2):
            ax.set_ylabel("Change in gap (pp)")
        if i in (2,3):
            ax.set_xlabel("Test year")
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=2,frameon=False,loc="lower center",
               bbox_to_anchor=(.5,.004))
    export(fig,"replay_contrasts")


def threshold_table(states,profiles,cuts,label):
    names=[STATE_NAMES[s] for s in states]
    lines=[r"\begin{table*}[!t]",
           rf"\caption{{Group Counts, Score Quartiles and Decision Thresholds in {names[0]} and {names[1]}}}\label{{{label}}}",
           r"\centering\footnotesize\setlength{\tabcolsep}{2pt}",
           r"\begin{tabular*}{.99\textwidth}{@{\extracolsep{\fill}}llrr rr rr rr rr@{}}\toprule",
           r" & & \multicolumn{2}{c}{Cohort size} & \multicolumn{2}{c}{Score quartile (25\%)} & \multicolumn{2}{c}{Pooled cutoff} & \multicolumn{2}{c}{Previous-year cutoff} & \multicolumn{2}{c}{Year-by-year cutoff} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}",
           r"State & Year & Black & White & Black & White & Black & White & Black & White & Black & White \\\midrule"]
    for state in states:
        if state!=states[0]:
            lines.append(r"\midrule")
        for y in YEARS:
            if y==2020:
                continue
            pair=profiles[(profiles.state==state)&(profiles.year==y)].set_index("group").loc[["Black","White"]]
            rows=cuts[(cuts.state==state)&(cuts.year==y)].set_index(["policy","group"])
            cells=[state if y==2012 else "",str(y)]
            cells.extend(f"{n:,}" for n in pair.n)
            cells.extend(f"{v:.4f}" for v in pair.p25)
            cells.extend(f"{rows.loc[(policy,group),'threshold']:.4f}" for policy in ["pooled","previous","common"] for group in ["Black","White"])
            lines.append(" & ".join(cells)+r" \\")
    lines.extend([r"\bottomrule\end{tabular*}\par\smallskip",
        r"\begin{minipage}{.99\textwidth}\footnotesize The score estimates the probability of income above the state's annual 60th percentile among eligible respondents. The screen targets a 75\% approval share and approves scores at or above the group cutoff. Earlier cohorts determine the strength used in each test. The five-point limit on the absolute Black--White approval gap applies to pooled history, the last available year or every prior year, respectively.\end{minipage}\end{table*}"])
    return lines


def main():
    cdfs=read(DIAG/"group_score_ecdf.csv")
    cuts=read(DIAG/"group_thresholds.csv")
    profiles=read(DIAG/"group_score_quantiles.csv")
    cohorts=read(COMP/"cohort_composition.csv")
    replay=read(TRANSFER/"model_cohort_replay.csv").query("policy=='common'")
    comps=read(COMP/"composition_contrasts.csv").query("policy=='common'")
    paired=profiles.merge(cohorts,on=["state","year","group"],validate="one_to_one")
    assert len(paired)==104 and (paired.n_x==paired.n_y).all()
    np.testing.assert_allclose(paired.group_share_pct,100*paired.group_share,atol=1e-12,rtol=0)
    relevant=cuts.query("group=='Black' and policy=='common'")[["state","year","gap_pp"]].merge(
        replay,on=["state","year"],validate="one_to_one").merge(
        comps.query("scenario=='observed'")[["state","year","gap_pp"]],on=["state","year"],validate="one_to_one")
    assert len(relevant)==48
    for name in ["gap_pp_x","gap_pp_y"]:
        np.testing.assert_allclose(relevant[name],relevant.target_gap_pp,atol=1e-11,rtol=0)
    diagnostic_figure(cdfs,cuts,replay,comps)
    profile_figure(profiles,cuts,cohorts)
    contrast_figure(replay,comps)
    parts=threshold_table(STATES[:2],profiles,cuts,"tab:scorecut_west")+[""]+threshold_table(
        STATES[2:],profiles,cuts,"tab:scorecut_east")
    (ROOT/"paper/generated/score_thresholds.tex").write_text("\n".join(parts)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
