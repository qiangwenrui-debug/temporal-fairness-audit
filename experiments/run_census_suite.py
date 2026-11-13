#!/usr/bin/env python3
"""
Runs the census ablations and checks (Tables V, VII, VIII, A3, A5) one after another, and collects the
outputs in census_suite/. Needs, in the same folder: run_temporal_audit.py,
delay_aware_monitor.py, monitor_ablations.py, audit_checks.py, robustness_checks.py and the
ACS files in acs_data/ (as downloaded by the earlier runs).

    python run_census_suite.py            # about 20-30 minutes on a laptop
    python run_census_suite.py --only monitor_CA_d1     # one job

Each job writes its own folder under census_suite/; a log of all jobs is written to
census_suite/run_log.txt.
"""
import argparse
import os
import subprocess
import sys
import time

INC = ["--task", "income", "--group", "race", "--label-mode", "quantile"]
JOBS = {
    # delay-aware monitoring: ablation (E3u), clipping, ESS by cell, policies with approval rates
    "monitor_CA_d1": ["monitor_ablations.py", *INC, "--states", "CA"],
    "monitor_CA_d1_hgb": ["monitor_ablations.py", *INC, "--states", "CA", "--domain-clf", "hgb"],
    "monitor_CA_d2": ["monitor_ablations.py", *INC, "--states", "CA", "--label-delay", "2"],
    "monitor_TX_d1": ["monitor_ablations.py", *INC, "--states", "TX"],
    "monitor_NY_d1": ["monitor_ablations.py", *INC, "--states", "NY"],
    "monitor_FL_d1": ["monitor_ablations.py", *INC, "--states", "FL"],
    "monitor_emp_race_CA_d1": ["monitor_ablations.py", "--task", "employment", "--group", "race",
                               "--states", "CA"],
    "monitor_inc_sex_CA_d1": ["monitor_ablations.py", "--task", "income", "--group", "sex",
                              "--label-mode", "quantile", "--states", "CA"],
    # audit protocol: pooled beta applied in every year, approval rates, fixed-budget beta*
    "audit_inc_race_CA": ["audit_checks.py", *INC, "--states", "CA"],
    "audit_inc_race_TX-NY-FL": ["audit_checks.py", *INC, "--states", "TX", "NY", "FL"],
    "audit_emp_race_CA": ["audit_checks.py", "--task", "employment", "--group", "race",
                          "--states", "CA"],
    "audit_inc_sex_CA": ["audit_checks.py", "--task", "income", "--group", "sex",
                         "--label-mode", "quantile", "--states", "CA"],
    "audit_emp_sex_CA": ["audit_checks.py", "--task", "employment", "--group", "sex",
                         "--states", "CA"],
    # robustness checks with the pooled gap of Proposition 1 and beta*_pool
    "robustness_CA": ["robustness_checks.py"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--data-dir", default="acs_data")
    a = ap.parse_args()
    root = "census_suite"
    os.makedirs(root, exist_ok=True)
    jobs = {k: v for k, v in JOBS.items() if a.only is None or k in a.only}
    t_all = time.time()
    with open(f"{root}/run_log.txt", "a") as log:
        for name, (script, *args) in jobs.items():
            out = f"{root}/{name}"
            cmd = [sys.executable, script, *args, "--data-dir", a.data_dir, "--out", out]
            print(f"[{time.time() - t_all:6.0f}s] {name} ...", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
            log.write(f"\n===== {name}: {status}, {time.time() - t0:.0f}s\n{' '.join(cmd)}\n")
            log.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
            log.flush()
            print(f"         {status} in {time.time() - t0:.0f}s", flush=True)
    print(f"Done in {time.time() - t_all:.0f}s. Results in {root}/")


if __name__ == "__main__":
    main()
