"""Run the official-ACS policy-monitor experiments and regenerate all monitoring reports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from policy_monitor_data import YEARS
from policy_monitor_summary import summarize

SCRIPTS = Path(__file__).resolve().parent
STATES = ("CA", "TX", "NY", "FL")
EXTENDED_YEARS = YEARS + [2024]


def run_job(root, out, name, options, threads):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(SCRIPTS/"policy_monitor.py"), "--data", str(root/"data"),
           "--out", str(out), "--threads", str(threads), *options]
    print("Running", name, flush=True)
    with (out/(name+".log")).open("w",encoding="utf8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Finished", name, flush=True)


def fits(root, workers, threads):
    out=root/"results/monitor"
    jobs=[(s+"_primary", ["--state",s,"--extend"]) for s in STATES]
    jobs += [("CA_employment", ["--state","CA","--task","employment"]),
             ("CA_real", ["--state","CA","--label","real"])]
    jobs += [(f"CA_seed{s}", ["--state","CA","--seed",str(s)]) for s in range(1,5)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(run_job,root,out,name,options,threads) for name,options in jobs]
        for future in pending:
            future.result()
    run_job(root,out,"CA_delay2",["--state","CA","--delay","2"],threads)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for tag,q in (("q060","0.60"),("q090","0.90")):
            for state in STATES:
                pending.append(pool.submit(run_job,root,out/tag,state,["--state",state,"--q",q,"--extend"],min(3,threads)))
        for future in pending:
            future.result()


def aggregate(root):
    out=root/"results/monitor"
    archive=root/"input_results/revision_results"
    summarize(out,archive,skip_reconciliation=not archive.is_dir())
    for tag in ("q060","q090"):
        summarize(out/tag,archive,skip_reconciliation=True)
    records=[json.loads((root/"data/normalized"/f"{s}_{y}.json").read_text(encoding="utf8")) for s in STATES for y in EXTENDED_YEARS]
    df=pd.DataFrame(records).drop(columns="columns").sort_values(["state","year"])
    df.to_csv(out/"data_manifest.csv",index=False)
    subprocess.run([sys.executable,str(SCRIPTS/"policy_monitor_report.py"),"--root",str(root)],check=True)


def check(root):
    out=root/"results/monitor"
    manifest=pd.read_csv(out/"data_manifest.csv")
    assert len(manifest)==40 and not manifest.duplicated(["state","year"]).any()
    for tag,q,expected in (("",.75,240),("q060",.60,96),("q090",.90,96)):
        path=out/tag
        yearly=pd.read_csv(path/"all_yearly.csv")
        assert len(yearly)==expected
        ids=["state","task","group","label","delay","seed","policy"]
        assert not yearly.duplicated(ids+["year"]).any()
        for _,part in yearly.groupby(ids):
            by_year=part.set_index("year")
            for row in part.itertuples():
                assert row.train_through<row.holdout<=row.year-row.delay
                if row.holdout in by_year.index:
                    assert np.isclose(row.E0,by_year.loc[row.holdout,"true_RG"])
                else:
                    assert pd.isna(row.E0)
                past=part.loc[part.year<=row.year-row.delay,"gamma_needed"]
                assert row.history_k==len(past)
                assert np.isclose(row.gamma_history,past.max(),equal_nan=True)
                if row.policy!="none":
                    assert abs(row.true_G)<=.050000001
                if row.policy=="per_period_budget":
                    assert abs(row.approval-q)<.0007
        intervals=pd.read_csv(path/"all_intervals.csv")
        assert (intervals.lo_pp<=intervals.hi_pp).all()
        flags=(intervals.lo_pp<=intervals.true_RG_pp) & (intervals.true_RG_pp<=intervals.hi_pp)
        assert (intervals.covered==flags).all()
        scores=pd.read_csv(path/"point_summary_common.csv")
        assert (scores.tp+scores.fp+scores.fn+scores.tn==scores.n).all()
    print("Verified raw-file manifest, expected rows, unique rule-years, E0 history, label timing, Gamma history, feasible policies, intervals and confusion totals.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=SCRIPTS.parents[1])
    p.add_argument("--stage",choices=["all","data","fits","aggregate","check"],default="all")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    a.root=a.root.resolve()
    if a.stage in ("all","data"):
        subprocess.run([sys.executable,str(SCRIPTS/"policy_monitor_data.py"),"--data-dir",str(a.root/"data"),
                        "--workers",str(a.workers),"--states",*STATES,"--years",*map(str,EXTENDED_YEARS)],check=True)
    if a.stage in ("all","fits"):
        fits(a.root,a.workers,a.threads)
    if a.stage in ("all","aggregate"):
        aggregate(a.root)
    if a.stage in ("all","check"):
        check(a.root)
