"""Paired annual validation, error counts, sensitivity decisions and snapshot reconciliation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ESTIMATORS = ["E0", "E1", "E2", "E3u", "E3"]
DETAIL = ["state","task","group","label","delay","seed","window","policy"]
BASE = ["task","group","label","delay","seed","window","policy"]


def error_summary(df):
    output = []
    for scope, keys in [("state",DETAIL),("states",BASE)]:
        for index, rows in df.groupby(keys):
            for e in ESTIMATORS:
                preds = rows[e].to_numpy()
                truth = rows.true_RG.to_numpy()
                valid = np.isfinite(preds)
                preds, truth = preds[valid], truth[valid]
                alert, real = abs(preds)>.02, abs(truth)>.02
                err = 100*(preds-truth)
                output.append(dict(scope=scope, **dict(zip(keys,index)), n=len(truth), states=rows.state.nunique(),
                    estimator=e, mae_pp=np.abs(err).mean(), bias_pp=err.mean(), max_ae_pp=np.abs(err).max(),
                    true_exceedances=int(real.sum()),tp=int((alert & real).sum()), fp=int((alert & ~real).sum()),
                    fn=int((~alert & real).sum()),tn=int((~alert & ~real).sum())))
    return pd.DataFrame(output)


def pair_table(common):
    pairs = []
    for other in ["E0","E2","E3u"]:
        d=common[DETAIL+["year","holdout","true_RG","E3","E3_ae_pp",other,other+"_ae_pp"]].copy()
        d["comparator"]=other
        d["comparator_ae_pp"]=d[other+"_ae_pp"]
        d["diff_ae_pp"]=d.E3_ae_pp-d.comparator_ae_pp
        pairs.append(d[DETAIL+["year","holdout","comparator","E3_ae_pp","comparator_ae_pp","diff_ae_pp"]])
    pairs = pd.concat(pairs,ignore_index=True)
    summary = []
    for scope, keys in [("state",DETAIL+["comparator"]),("states",BASE+["comparator"])]:
        for index,r in pairs.groupby(keys):
            yearly = r.groupby("year").diff_ae_pp.mean()
            leave = [(yearly.sum()-v)/(len(yearly)-1) for v in yearly] if len(yearly)>1 else [np.nan]
            summary.append(dict(scope=scope,**dict(zip(keys,index)),n=len(r),years=len(yearly),win_E3=int((r.diff_ae_pp<0).sum()),
                 losses_E3=int((r.diff_ae_pp>0).sum()),mean_diff_pp=r.diff_ae_pp.mean(),
                 min_leave_one_calendar_year_pp=min(leave),max_leave_one_calendar_year_pp=max(leave)))
    return pairs,pd.DataFrame(summary)


def decide_summary(table):
    out=[]
    for scope,keys in [("state",DETAIL+["gamma_method"]),("states",BASE+["gamma_method"])]:
        for index,r in table.groupby(keys):
            out.append(dict(scope=scope,**dict(zip(keys,index)),n=len(r), covered=int(r.covered.sum()),
                mean_width_pp=r.width_pp.mean(), definite_outside=int((r.action=="outside").sum()),
                definite_inside=int((r.action=="inside").sum()), abstain=int((r.action=="abstain").sum()),
                wrong_definite=int(r.incorrect_definite.sum()), true_outside=int((r.actual=="outside").sum()),
                false_clearance=int(((r.action=="inside") & (r.actual=="outside")).sum()),
                false_alarm=int(((r.action=="outside") & (r.actual=="inside")).sum())))
    return pd.DataFrame(out)


def reconcile(yearly, archive):
    specs = [("CA","income",1,"monitor_CA_d1"),("CA","income",2,"monitor_CA_d2"),
              ("CA","employment",1,"monitor_emp_race_CA_d1")]
    specs += [(st,"income",1,f"monitor_{st}_d1") for st in ("TX","NY","FL")]
    records=[]
    for st,task,lag,dirname in specs:
        curr=yearly.query("state==@st and task==@task and delay==@lag and seed==0 and window=='original_years' and group=='race' and policy=='none' and label!='real'").copy()
        if curr.empty:
            continue
        orig=pd.read_csv(archive/dirname/"estimates.csv").set_index("year")
        for _,v in curr.iterrows():
            old=orig.loc[int(v.year)]
            if v.nH!=old.n_H or v.holdout!=old.holdout:
                raise ValueError(f"Input cohort mismatch: {st} {task} {lag} {v.year}")
            # The archived column E0 uses the CURRENT model rescoring H with own thresholds
            # (E0 prime). Recover the actual deployed stale residual from archived H's truth.
            for e in ["true_G","true_RG","E0","E1","E2","E3u","E3"]:
                archive_q= (orig.loc[int(v.holdout),"true_RG"] if int(v.holdout) in orig.index else np.nan) if e=="E0" else old[e]
                records.append(dict(state=st,task=task,delay=lag,year=v.year,quantity=e,archive_pp=100*archive_q,
                                    rerun_pp=100*v[e],delta_pp=100*(v[e]-archive_q)))
    return pd.DataFrame(records)


def summarize(root, archive, skip_reconciliation=False):
    yearly=pd.concat([pd.read_csv(f) for f in sorted(root.glob("*/yearly.csv"))],ignore_index=True)
    intervals=pd.concat([pd.read_csv(f) for f in sorted(root.glob("*/intervals.csv"))],ignore_index=True)
    if yearly.duplicated(DETAIL+["year"]).any(): raise ValueError("duplicated state–configuration–year")
    common=yearly.loc[yearly.E0.notna()].copy()
    # Fixed and learned Gamma compared on exactly the historical-Gamma-available years.
    matched=intervals[intervals.gamma_method=="history"][DETAIL+["year"]]
    common_iv=intervals.merge(matched,on=DETAIL+["year"],how="inner",validate="many_to_one")
    pairs,pair_sum=pair_table(common)
    tables={"all_yearly":yearly,"point_summary_common":error_summary(common),"pairwise_years":pairs,
            "paired_summary_common":pair_sum,"all_intervals":intervals,"interval_summary_common":decide_summary(common_iv),
            }
    if not skip_reconciliation:
        tables["reconciliation"]=reconcile(yearly,archive)
    for name,table in tables.items():
        tmp=root/(name+".tmp")
        table.to_csv(tmp,index=False)
        tmp.replace(root/(name+".csv"))
    print(f"{len(yearly)} policy-years; {yearly[DETAIL[:6]].drop_duplicates().shape[0]} runs; {len(common)} common policy-years",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--skip-reconciliation",action="store_true",help="for alternate q values without corresponding archived monitoring data")
    args=p.parse_args();summarize(args.out,args.archive,args.skip_reconciliation)
