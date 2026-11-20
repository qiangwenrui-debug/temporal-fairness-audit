"""Generate supplementary monitoring tables directly from the paired annual outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def write_table(rows, caption, label, spec, heads, body, note):
    rows.extend([r"\begin{table*}[!t]", r"\caption{"+caption+r"}\label{"+label+"}",
                 r"\centering\footnotesize",r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{"+spec+"}",r"\toprule", " & ".join(heads)+r" \\",r"\midrule"])
    for line in body:
        rows.append(" & ".join(str(word) for word in line)+r" \\")
    rows.extend([r"\bottomrule",r"\end{tabular}",r"\par\vspace{3pt}\begin{minipage}{0.96\textwidth}\footnotesize "+note+r"\end{minipage}",
                 r"\end{table*}", ""])


def main(root):
    dest=root/"paper/generated"; dest.mkdir(exist_ok=True, parents=True)
    disk=root/"results/monitor"
    lines=[]
    agency={"none":"Unadjusted","per_period":"Annual minimum","per_period_budget":"Fixed budget"}
    folders={.60:disk/"q060",.75:disk,.90:disk/"q090"}
    scores={q:pd.read_csv(path/"point_summary_common.csv") for q,path in folders.items()}
    pairs={q:pd.read_csv(path/"paired_summary_common.csv") for q,path in folders.items()}
    years={}
    for q,path in folders.items():
        annual=pd.read_csv(path/"all_yearly.csv")
        for _,runs in annual.groupby(["state","task","group","label","delay","seed","policy"]):
            if not runs.E0.notna().equals(runs.holdout.isin(runs.year)):
                raise ValueError("E0 availability disagrees with the dated deployment history")
        years[q]=annual.loc[annual.E0.notna()]
        if not np.isfinite(years[q][["true_RG","E0","E1","E2","E3u","E3"]].to_numpy()).all():
            raise ValueError("Incomplete common monitoring window")
    def base(df,scope="states"):
        return df.query("scope==@scope and label=='quantile' and task=='income' and group=='race' and delay==1 and seed==0")
    def fp_fn(s):
        fp=str(s.fp) if s.n>s.true_exceedances else "---"
        fn=str(s.fn) if s.true_exceedances else "---"
        return f"({fp}, {fn})"
    body=[]
    for window, tag in [("original_years","2017--23"),("external_2024","2024")]:
        for q in (.60,.90):
            df=base(scores[q]).query("window==@window")
            frontier=years[q].query("window==@window and group=='race' and task=='income' and label=='quantile' and delay==1 and seed==0")
            q_rows=[]
            for policy,name in agency.items():
                vals=df.query("policy==@policy").set_index("estimator")
                e2,e3=vals.loc["E2"],vals.loc["E3"]
                selected=frontier.loc[frontier.policy==policy]
                n=24 if window=="original_years" else 4
                if len(selected)!=n or selected.duplicated(["state","year"]).any() or e3.n!=n:
                    raise ValueError(f"Incomplete state-year panel for {q}, {policy}, {window}")
                if (selected.true_RG.abs()>.02).sum()!=e3.true_exceedances:
                    raise ValueError("Annual event count differs from table summary")
                q_rows.append([tag,f"{q:.2f}",name,f"{e3.true_exceedances}/{e3.n}",
                               f"{100*selected.true_RG.abs().max():.2f}",f"{e2.mae_pp:.2f}",fp_fn(e2),f"{e3.mae_pp:.2f}",fp_fn(e3)])
            if all(r[3:]==q_rows[0][3:] for r in q_rows):
                body.append([tag,f"{q:.2f}","All three",*q_rows[0][3:]])
            else:
                body.extend(q_rows)
    hist60=years[.60].query("policy=='per_period' and window=='original_years'").sort_values(["state","year"])
    events=hist60[hist60.true_RG.abs()>.02]
    cases=", ".join("%s %d ($%+.2f$ pp)" % (r.state,r.year,100*r.true_RG) for r in events.itertuples())
    hits=", ".join(f"{r.state} {r.year}" for r in events.loc[events.E3.abs()>.02].itertuples())
    iv=pd.read_csv(folders[.60]/"interval_summary_common.csv")
    hist=base(iv).query("policy=='per_period' and gamma_method=='history'").set_index("window")
    pre,late=hist.loc["original_years"],hist.loc["external_2024"]
    pressure=(f"At $q=0.60$, historical events after annual calibration were {cases}. The covariate-weighted reference estimator detected {hits}. "
              f"Historical $\\Gamma$ yielded {pre.definite_outside} definite alert, {pre.definite_inside} in-band decisions "
              f"and {pre.abstain} unresolved decisions. Coverage was {pre.covered}/{pre.n} historically and {late.covered}/{late.n} in 2024, "
              f"when {late.abstain} decisions were unresolved.")
    write_table(lines,"Residual Monitoring at Two Approval Shares in Four States",
        "tab:monitor-q-stress","@{}lllrccccc@{}",
        ["Window",r"$q$","Rule","Events/$n$",r"Max.\ $|RG_t|$",r"\shortstack{Recalibrated\\MAE}",r"\shortstack{Recalibrated\\(FP, FN)}",r"\shortstack{Weighted\\MAE}",r"\shortstack{Weighted\\(FP, FN)}"],body,
        r"$q$ is the target approval share. An event occurs when the absolute Black--White approval gap among respondents above the state-specific annual income cutoff exceeds two percentage points (pp). The four-state models use training seed 0 and a one-year label delay. The two estimators use recalibrated scores and the covariate-weighted reference cohort, respectively. Mean absolute error (MAE) and the maximum absolute residual are in pp. Counts give false alerts (FP) and missed events (FN). A dash marks an empty comparison group. Each state contributes 2017--2019, 2021--2023 and one later observation (2024). The rules share a model and weights within each state-year. At $q=0.90$ all three rules have identical counts and agree on the displayed errors and maximum to two decimal places. "+pressure)
    body=[]
    for task,label,lag,seed,desc in [("income","quantile",1,z,"Income, annual quantile") for z in range(5)]+[
            ("income","real",1,0,r"Income, real \$50,000"),("income","quantile",2,0,"Income, annual quantile"),
            ("employment","employed",1,0,"Employment")]:
        df=scores[.75].query("state=='CA' and task==@task and label==@label and delay==@lag and seed==@seed and window=='original_years'")
        base_e=df.query("policy=='none' and estimator=='E3'").iloc[0]
        vals=df.query("policy=='per_period'").set_index("estimator")
        e2,e3=vals.loc["E2"],vals.loc["E3"]
        state_years=years[.75].query("state=='CA' and group=='race' and task==@task and label==@label and delay==@lag and seed==@seed and window=='original_years' and policy=='per_period'")
        if len(state_years)!=e3.n or e3.true_exceedances or (state_years.true_RG.abs()>.02).any():
            raise ValueError("Update the no-event note for the California robustness comparisons")
        body.append([desc,lag,seed,f"{base_e.true_exceedances}/{e3.n}",f"{100*state_years.true_RG.abs().max():.2f}",
                     f"{base_e.mae_pp:.2f}",f"{e2.mae_pp:.2f}",f"{e3.mae_pp:.2f}"])
    write_table(lines,"California Monitoring Results by Seed, Outcome Definition and Label Delay",
        "tab:monitor-seeds-controls","@{}lrrrrrrr@{}",
        ["Task and label","Delay","Seed",r"\shortstack{Unadjusted\\events/$n$}",r"\shortstack{Annual\\max.\ $|RG_t|$}",r"\shortstack{Unadjusted\\weighted MAE}",r"\shortstack{Annual\\recalibrated MAE}",r"\shortstack{Annual\\weighted MAE}"],body,
        r"An event is a Black--White approval gap greater than two percentage points in absolute value among respondents with a favourable outcome. Both rules target a 75\% approval share. Mean absolute error (MAE) is reported in percentage points for recalibrated scores and for the covariate-weighted reference estimate. All annually calibrated gaps were within the two-point band. Each comparison uses years with a previously observed residual. Seeds govern capped subsampling, XGBoost and the domain-classifier folds on the same calendar years. The real-dollar cutoff uses 2015 prices. Employment uses the race groups and the Folktables employment outcome. With two-year delay, the common years are 2019 and 2021--2023.")
    body=[]
    for q in (.60,.75,.90):
        df=base(pairs[q]).query("policy=='per_period'")
        a=df.query("window=='original_years'").set_index("comparator")
        b=df.query("window=='external_2024'").set_index("comparator")
        for e in ("E0","E2","E3u"):
            x=a.loc[e]; y=b.loc[e]
            def signed(value):
                digits=4 if 0 < abs(value) < .0005 else 3
                return f"{value:+.{digits}f}"
            if x.win_E3+x.losses_E3!=x.n or y.win_E3+y.losses_E3!=y.n:
                raise ValueError("Tied state-years require a separate count")
            comparator = {"E0": "Last observed residual", "E2": "Recalibrated scores", "E3u": "Unweighted reference"}[e]
            body.append([f"{q:.2f}",comparator,signed(x.mean_diff_pp),f"{x.win_E3}/{x.losses_E3}",
                         f"[{signed(x.min_leave_one_calendar_year_pp)}, {signed(x.max_leave_one_calendar_year_pp)}]",
                         signed(y.mean_diff_pp),f"{y.win_E3}/{y.losses_E3}"])
    write_table(lines,"Paired Differences in Residual Estimation Error After Annual Calibration",
        "tab:monitor-paired-differences","@{}llrcrrc@{}",
        [r"$q$","Comparator",r"\shortstack{Historical\\difference}",r"\shortstack{Weighted error\\lower/higher}",r"\shortstack{Omit one year\\(range)}",r"\shortstack{2024\\difference}",r"\shortstack{Weighted error\\lower/higher}"],body,
        r"For the Black--White approval gap among respondents above the annual income cutoff, differences are the weighted reference estimate's absolute error minus the comparator's, in percentage points. Negative differences favour weighting. Lower/higher counts state-years in each direction. At each target approval share $q$, the historical mean uses six test years per state (2017--2019 and 2021--2023). The 2024 mean uses the same four states. To show how much the historical mean depends on any one calendar year, we omit that year from all four states, recalculate the mean and report the range across omissions.")
    (dest/"stress.tex").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    main(p.parse_args().root)
