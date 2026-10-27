#!/usr/bin/env python3
"""Prints the numbers missing from Tables III and IV of the draft.
Run in the folder that contains the results_* folders:  python collect_numbers.py
"""
import os
import pandas as pd

SETTINGS = [("Income x race, CA", "results_income_race_CA_d1_quantile"),
            ("Income x race, TX/NY/FL", "results_income_race_TX-NY-FL_d1_quantile"),
            ("Employment x race, CA", "results_employment_race_CA_d1"),
            ("Income x sex, CA", "results_income_sex_CA_d1_quantile"),
            ("Employment x sex, CA", "results_employment_sex_CA_d1")]

print("=== Table III: individuals per test year (XGBoost rows, q = 0.75) ===")
for name, folder in SETTINGS:
    f = os.path.join(folder, "yearly.csv")
    if not os.path.exists(f):
        print(f"{name}: MISSING {f}")
        continue
    y = pd.read_csv(f)
    y = y[(y.model == "xgb") & (y.q == 0.75)]
    print(f"{name}: n = {int(y.n.min()):,}-{int(y.n.max()):,} "
          f"(group 1: {int(y.n1.min()):,}-{int(y.n1.max()):,}); "
          f"favourable rate g1 {100*y.pi1.min():.1f}-{100*y.pi1.max():.1f}%, "
          f"g0 {100*y.pi0.min():.1f}-{100*y.pi0.max():.1f}%")

print("\n=== Table IV: employment x sex control row (q = 0.75) ===")
folder = "results_employment_sex_CA_d1"
p, c = os.path.join(folder, "pooled.csv"), os.path.join(folder, "carryover.csv")
if os.path.exists(p):
    P = pd.read_csv(p)
    C = pd.read_csv(c)
    for m in ["lr", "xgb", "lgbm"]:
        r = P[(P.model == m) & (P.q == 0.75)].iloc[0]
        cc = C[(C.model == m) & (C.q == 0.75)]
        out = cc[cc.outside]
        largest = f"{100*out.gap.abs().max():.1f}" if len(out) else "-"
        print(f"{m}: pooled gap {100*r.pooled_gap:+.1f} pp [{100*r.pooled_lo:+.1f}, {100*r.pooled_hi:+.1f}]; "
              f"mean|G| {100*r.mean_abs_yearly:.1f}; outside +-5: {int(r.yrs_out_5)}/{int(r.n_years)}; "
              f"beta* {r.beta_min:.2f}-{r.beta_max:.2f}; carry-over {int(cc.outside.sum())}/{len(cc)} ({largest}); "
              f"yearly G {100*r.min_yearly:+.1f} to {100*r.max_yearly:+.1f}")
    pb = pd.read_csv(os.path.join(folder, "pooled_beta_by_q.csv"))
    print("pooled beta* at q=0.75:", pb[pb.q == 0.75][["model", "pooled_beta_star"]].to_dict("records"))
else:
    print(f"MISSING {p}: run  python run_temporal_audit.py --states CA --group sex --task employment")

print("\n=== Carry-over failure years (for the [VERIFY] in Section VI-D) ===")
for name, folder in SETTINGS:
    f = os.path.join(folder, "carryover.csv")
    if os.path.exists(f):
        C = pd.read_csv(f)
        C = C[(C.model == "xgb") & (C.q == 0.75) & (C.outside)]
        print(f"{name}: " + ", ".join(f"{int(r.year)} ({100*r.gap:+.1f})" for r in C.itertuples()))
