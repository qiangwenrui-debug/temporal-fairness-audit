"""Render origin composition and race-definition comparisons for the 2021 frozen strength."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from revision_figures import BLUE, INK, MUTED, ORANGE, STATES, TEAL, WIDTH


ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = {
    "alone": ("Race alone", INK, "o"),
    "including_combinations": ("Including combinations", BLUE, "s"),
    "alone_nonhispanic": ("Non-Hispanic, race alone", TEAL, "^"),
}


def read_inputs():
    folder = ROOT/"results/diagnostics"
    replay = pd.read_csv(folder/"origin_replay.csv").query("year==2021 and policy=='common'")
    ci = pd.read_csv(folder/"origin_intervals.csv").query("metric=='change_pp' and policy=='common'")
    rows = replay.merge(ci[["state","definition","estimate_pp","lower_pp","upper_pp","draws"]],
                        on=["state","definition"],validate="one_to_one")
    if (len(rows)!=12 or set(rows.definition)!=set(DEFINITIONS)
            or not rows.draws.eq(1000).all() or not rows.reference_year.eq(2019).all()):
        raise ValueError("Incomplete paired household results for four states and three definitions.")
    np.testing.assert_allclose(rows.change_pp,rows.estimate_pp,rtol=0,atol=1e-10)
    primary = rows[rows.definition=="alone"]
    np.testing.assert_allclose(primary.change_pp,primary.origin_mix_pp+primary.within_origin_pp,
                               rtol=0,atol=1e-10)
    base = pd.read_csv(folder/"eligible_race_origin.csv").query("year in [2019,2021] and rac1p in [1,2]")
    n = base.pivot(index=["state","year","rac1p"],columns="hispanic",values="n")
    if len(n)!=16 or n.isna().any().any():
        raise ValueError("Missing race-by-origin denominator.")
    cells = pd.read_csv(folder/"origin_cells.csv").query("year==2021 and policy=='common' and definition=='alone'").copy()
    cells["rac1p"] = cells.black+1
    for year,col in ((2019,"prior_n"),(2021,"n")):
        observed = cells.pivot(index=["state","rac1p"],columns="hispanic",values=col)
        expected = n.xs(year,level="year")
        np.testing.assert_array_equal(observed.loc[expected.index,expected.columns],expected)
    return rows.set_index(["state","definition"]),100*n[1]/n.sum(axis=1)


def figure(rows, share):
    fig,(left,right)=plt.subplots(1,2,figsize=(WIDTH,2.90),gridspec_kw={"width_ratios":[1,1.13]})
    fig.subplots_adjust(left=.072,right=.982,top=.865,bottom=.30,wspace=.23)
    levels=np.arange(3,-1,-1)
    for i,state in enumerate(STATES):
        y=levels[i]
        for group,color,dy in [(2,BLUE,.12),(1,ORANGE,-.12)]:
            old,new=[share.loc[(state,year,group)] for year in (2019,2021)]
            left.plot([old,new],[y+dy,y+dy],color=color,lw=1.1,ls="-" if group==2 else "--")
            left.plot(old,y+dy,ls="none",marker="o",mfc="white",mec=color,mew=1.1,ms=5.1)
            left.plot(new,y+dy,ls="none",marker="o",color=color,ms=4.9)
        for dy,(key,(_,color,mark)) in zip((.22,0,-.22),DEFINITIONS.items()):
            row=rows.loc[(state,key)]
            right.hlines(y+dy,row.lower_pp,row.upper_pp,color=color,lw=.95)
            right.vlines([row.lower_pp,row.upper_pp],y+dy-.045,y+dy+.045,color=color,lw=.75)
            right.plot(row.change_pp,y+dy,ls="none",marker=mark,color=color,ms=4.7)
    for ax in (left,right):
        ax.set_yticks(levels,STATES)
        ax.set_ylim(-.43,3.45)
    left.set_title("(a) Hispanic origin within each race-alone group",loc="left",fontsize=8.2,pad=5)
    left.set_xlabel("Hispanic share of group (%)")
    left.set_xlim(-.5,34.5)
    left.set_xticks([0,10,20,30])
    handles=[Line2D([],[],color=BLUE,label="Black alone"),
             Line2D([],[],color=ORANGE,ls="--",label="White alone"),
             Line2D([],[],marker="o",color=INK,mfc="white",ls="none",label="2019",ms=4.5),
             Line2D([],[],marker="o",color=INK,ls="none",label="2021",ms=4.5)]
    left.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.30),ncol=2,
                frameon=False,columnspacing=1.5,handlelength=1.8,fontsize=7)
    right.set_title("(b) Change in gap by group definition",loc="left",fontsize=8.2,pad=5)
    right.set_xlabel("2021 minus 2019 gap (pp)")
    right.axvline(0,color=MUTED,ls=(0,(2.5,2)),lw=.8,zorder=0)
    right.set_xlim(-7.8,2.1)
    right.set_xticks([-6,-4,-2,0,2])
    handles=[Line2D([],[],color=color,marker=mark,ms=4.5,lw=.9,label=label)
             for label,color,mark in DEFINITIONS.values()]
    right.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,-.30),ncol=2,
                 frameon=False,columnspacing=.65,handlelength=.9,handletextpad=.45,fontsize=6.55)
    out=ROOT/"paper/figures"
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/"origin_reporting.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(out/"origin_reporting.png",dpi=360)
    plt.close(fig)


def table(rows):
    lines=[r"\begin{table*}[!t]",
           r"\caption{Approval Gaps by Race Definition and Origin Composition in 2019 and 2021}\label{tab:origindiag}",
           r"\centering\footnotesize\setlength{\tabcolsep}{2.3pt}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}\toprule",
           r" & & \multicolumn{2}{c}{Race alone} & \multicolumn{2}{c}{Change, race alone} & \multicolumn{2}{c}{Including combinations} & \multicolumn{2}{c}{Non-Hispanic, race alone} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
           r"State & $\beta$ & 2019 & 2021 & Origin mix & Within origin & 2019 & 2021 & 2019 & 2021 \\\midrule"]
    for s in STATES:
        a=rows.loc[(s,"alone")]
        cells=[s,f"{a.beta:.3f}"]
        values=[a.prior_gap_pp,a.gap_pp,a.origin_mix_pp,a.within_origin_pp]
        for key in ("including_combinations","alone_nonhispanic"):
            rec=rows.loc[(s,key)]
            values.extend([rec.prior_gap_pp,rec.gap_pp])
        cells.extend(f"${v:+.3f}$" if abs(abs(v)-5)<.005 else f"${v:+.2f}$" for v in values)
        lines.append(" & ".join(cells)+r" \\")
    lines.extend([r"\bottomrule\end{tabular*}\par\smallskip",
                  r"\begin{minipage}{.99\textwidth}\footnotesize Gaps are Black minus White approval rates in percentage points. The year-by-year rule uses the same strength, selected before 2021, in both years. We recompute score quantiles for each year and group definition. The two middle terms decompose the race-alone change into changes in Hispanic-origin shares within each group and changes in approval rates within each origin cell, averaging the endpoint rates and shares, respectively. Including combinations adds respondents who report either Black or White together with another race. Respondents reporting both Black and White are excluded. The final columns restrict both race-alone groups to non-Hispanic respondents.\end{minipage}\end{table*}"])
    (ROOT/"paper/generated/origin_diagnostics.tex").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    rows,share=read_inputs()
    figure(rows,share)
    table(rows)
