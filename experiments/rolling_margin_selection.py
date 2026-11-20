"""Select historical headroom from completed same-state forward tests, then score the next cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extended_audit_scores import EARLY, LATER, ROOT, STATES
from historical_margin_sensitivity import HISTORICAL_TOLERANCES
from policy_monitor_data import FIPS
from rolling_transfer import resampled_intervals
from run_temporal_audit import BETAS
from transfer_checks import gaps, household_codes, score_curves, smallest


MARGINS = (0., .5, 1.)
FIXED = ("fixed_0", "fixed_0_5", "fixed_1")


def calendar_and_paths(root, state, origin):
    if origin == 2010:
        years = EARLY + LATER
        data = root / "data/early_rolling"
        result = root / "results/transfer/from2010"
    else:
        years = LATER
        data = root / "data"
        result = root / "results/transfer"
    directory = data / "scores/audit" / f"{state}_income_race_quantile_d1_s0"
    # The first model-training year has no test scores. The next year seeds the calibration history.
    return years[1:], directory, result


def choices_before_test(year, gap_history, completed):
    """Read only earlier state-specific batches and their previously issued candidate decisions."""
    if any(y >= year for y in gap_history) or any(r["year"] >= year for r in completed):
        raise ValueError("The test cohort has entered the selection history")
    years = tuple(gap_history)
    inner = years[1:]
    worst = np.max(np.abs(np.stack(list(gap_history.values()))), axis=0)
    selections = []
    for margin, tolerance, policy in zip(MARGINS, HISTORICAL_TOLERANCES, FIXED, strict=True):
        past = [r for r in completed if r["policy"] == policy]
        if tuple(r["year"] for r in past) != inner:
            raise ValueError(f"Incomplete rolling validation history for {policy}")
        idx = smallest(worst <= tolerance)
        selections.append(dict(margin_pp=margin, grid_index=idx, beta=BETAS[idx],
                               worst_past_gap_pp=100*worst[idx],
                               prior_breaches=sum(not r["within"] for r in past),
                               prior_changed_vs_common=sum(r["changed_vs_common"] for r in past),
                               inner_n_people=sum(r["n"] for r in past)))
    chosen = min(selections, key=lambda r:(r["prior_breaches"], r["prior_changed_vs_common"], r["margin_pp"]))
    return chosen, selections


def audit_state(root, state, origin, last_year=2024):
    years, score_dir, _ = calendar_and_paths(root, state, origin)
    history, completed, yearly, selections = {}, [], [], []
    for year in (y for y in years if y <= last_year):
        if history:
            # Freeze both the margin and its historical strength before opening the test score file.
            choice, candidates = choices_before_test(year, history, completed)
            calendar = dict(state=state, year=year, history_years="|".join(map(str,history)),
                            inner_years="|".join(map(str,tuple(history)[1:])), n_inner=len(history)-1)
            for c in candidates:
                selections.append(dict(**calendar, **c, chosen=c["margin_pp"]==choice["margin_pp"]))
        with np.load(score_dir / f"{year}.npz") as scores:
            score = scores["s"].astype(float)
            group = scores["g"]
        sizes, curves = score_curves(score, group, .75, with_budget=False)
        approvals = curves[("probability","interpolated")]
        curve = gaps(approvals, sizes)
        if history:
            pooled = np.quantile(score,.25)
            anchors = np.array([np.quantile(score[group==g],.25) for g in (0,1)])
            cutoffs = [(1-c["beta"])*pooled + c["beta"]*anchors for c in candidates]
            decisions = [score >= threshold[group] for threshold in cutoffs]
            baseline = decisions[0]
            base_inside = bool(abs(curve[candidates[0]["grid_index"]]) <= .05)
            new = []
            for policy, c, cut, d in zip(FIXED, candidates, cutoffs, decisions, strict=True):
                counts = np.bincount(group[d],minlength=2)
                k = c["grid_index"]
                np.testing.assert_array_equal(counts,approvals[k])
                inside = bool(abs(curve[k]) <= .05)
                added = int(np.count_nonzero(d & ~baseline))
                removed = int(np.count_nonzero(~d & baseline))
                row = dict(**calendar, policy=policy, margin_pp=c["margin_pp"],
                           historical_tolerance_pp=5-c["margin_pp"], test_tolerance_pp=5.,
                           beta=c["beta"], tau0=cut[0], tau1=cut[1], n=len(group),
                           n0=int(sizes[0]), n1=int(sizes[1]), approved0=int(counts[0]), approved1=int(counts[1]),
                           gap_pp=100*curve[k], within=inside, baseline_within=base_inside,
                           newly_within=inside and not base_inside, newly_outside=base_inside and not inside,
                           changed_vs_common=added+removed, added_vs_common=added, removed_vs_common=removed,
                           changed_vs_fixed_0_5=int(np.count_nonzero(d!=decisions[1])),
                           changed_vs_fixed_1=int(np.count_nonzero(d!=decisions[2])))
                new.append(row)
            rolling = next(r for r in new if r["margin_pp"]==choice["margin_pp"]).copy()
            rolling["policy"] = "rolling"
            yearly.extend(new + [rolling])
            completed.extend(new)
        history[year] = curve
    return yearly, selections


def verify(yearly, selections, root, origin):
    years, _, dest = calendar_and_paths(root, STATES[0], origin)
    keys = ["state","year","historical_tolerance_pp"]
    fixed = yearly[yearly.policy != "rolling"].set_index(keys).sort_index()
    if origin == 2010:
        original = pd.read_csv(dest/"historical_margin_sensitivity.csv",
                               float_precision="round_trip").set_index(keys).sort_index()
        assert original.index.equals(fixed.index)
        for field in ("n","n0","n1","approved0","approved1","within","newly_within","newly_outside",
                      "changed_vs_common","added_vs_common","removed_vs_common"):
            np.testing.assert_array_equal(fixed[field],original[field])
        for field in ("gap_pp","beta","tau0","tau1"):
            np.testing.assert_allclose(fixed[field],original[field],atol=1e-11,rtol=0)
    archived = pd.read_csv(dest/"rolling_state_year.csv",float_precision="round_trip")
    archived = archived[archived.policy=="common"].set_index(keys[:2]).sort_index()
    zero = yearly[yearly.policy=="fixed_0"].set_index(keys[:2]).sort_index()
    assert archived.index.equals(zero.index)
    for field in ("n0","n1","approved0","approved1","gap_pp","beta","within"):
        np.testing.assert_allclose(zero[field],archived[field],atol=1e-11,rtol=0)
    assert len(yearly)==len(STATES)*(len(years)-1)*4 and len(selections)==len(yearly)*3//4
    # Independent audit of the ledger, rebuilding each inner loss directly from the recorded
    # group totals instead of using the pass/fail flags or the selector's running history.
    years = years[1:]
    for state in STATES:
        state_folds = fixed.reset_index().query("state==@state")
        ledger = selections[selections.state == state]
        for i, year in enumerate(years):
            past = state_folds[state_folds.year < year].copy()
            breach_num = (past.approved1*past.n0 - past.approved0*past.n1).abs()
            past["breached"] = 20*breach_num > past.n1*past.n0
            objectives = []
            for margin in MARGINS:
                rows = past[past.margin_pp==margin]
                objective = (int(rows.breached.sum()),int(rows.changed_vs_common.sum()),margin)
                objectives.append(objective)
                l = ledger[(ledger.year==year)&(ledger.margin_pp==margin)].iloc[0]
                assert (l.prior_breaches,l.prior_changed_vs_common,l.margin_pp) == objective
                assert l.n_inner==i and l.inner_years=="|".join(map(str,years[:i]))
            chosen = ledger[(ledger.year==year)&ledger.chosen]
            assert len(chosen)==1 and chosen.iloc[0].margin_pp == min(objectives)[2]
    # Repeat a truncated backtest. Removing all subsequent score files from its input
    # calendar must preserve every earlier candidate, action and inner audit field.
    for state in STATES:
        prefix, log = audit_state(root,state,origin,last_year=2019)
        reference = yearly[(yearly.state==state)&(yearly.year<=2019)].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.DataFrame(prefix),reference)
        reference = selections[(selections.state==state)&(selections.year<=2019)].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.DataFrame(log),reference)


def summarize(yearly):
    rows = []
    for state in (*STATES,"All"):
        block = yearly if state=="All" else yearly[yearly.state==state]
        n_breaches = {name:int((~data.within).sum()) for name,data in block.groupby("policy")}
        for policy in (*FIXED,"rolling"):
            data = block[block.policy == policy]
            breaches = n_breaches[policy]
            rows.append(dict(state=state,policy=policy,n_folds=len(data),n_people=int(data.n.sum()),
                             within=int(data.within.sum()),breaches=breaches,
                             breaches_avoided=n_breaches["fixed_0"]-breaches,
                             rescued=int(data.newly_within.sum()),harmed=int(data.newly_outside.sum()),
                             changed_vs_common=int(data.changed_vs_common.sum()),
                             added_vs_common=int(data.added_vs_common.sum()),
                             removed_vs_common=int(data.removed_vs_common.sum()),
                             changed_vs_fixed_0_5=int(data.changed_vs_fixed_0_5.sum()),
                             changed_vs_fixed_1=int(data.changed_vs_fixed_1.sum()),
                             breaches_avoided_vs_fixed_0_5=n_breaches["fixed_0_5"]-breaches,
                             breaches_avoided_vs_fixed_1=n_breaches["fixed_1"]-breaches))
    return pd.DataFrame(rows)


def intervals_2021(root, yearly, origin):
    """Paired pointwise intervals holding the prior model and both levels of selection fixed."""
    rows = []
    for state in STATES:
        _, score_dir, dest = calendar_and_paths(root,state,origin)
        subset = yearly[(yearly.state==state)&(yearly.year==2021)].copy()
        with np.load(score_dir/"2021.npz") as data:
            s,g,y = data["s"].astype(float),data["g"],data["y"]
        hh = household_codes(root/f"data/normalized/{state}_2021.parquet",g,y)
        seed = 32024 + 100*(2021-2024) + FIPS[state]
        lo,hi = resampled_intervals(s,g,hh,subset.beta.to_numpy(),seed,1000)
        subset["lower_pp"],subset["upper_pp"] = lo,hi
        archived = pd.read_csv(dest/"rolling_state_year.csv",float_precision="round_trip")
        old = archived[(archived.state==state)&(archived.year==2021)&(archived.policy=="common")]
        np.testing.assert_allclose(subset.loc[subset.policy=="fixed_0",["lower_pp","upper_pp"]].values,
                                   old[["lower_pp","upper_pp"]].values,atol=1e-11,rtol=0)
        keep = ["state","year","policy","margin_pp","beta","gap_pp","lower_pp","upper_pp"]
        rows.append(subset[keep].assign(draws=1000,seed=seed,n_households=int(hh.max())+1))
    return pd.concat(rows,ignore_index=True)


def main(root, origin):
    yearly, selections = [], []
    for state in STATES:
        rows, log = audit_state(root,state,origin)
        yearly.extend(rows)
        selections.extend(log)
    yearly, selections = pd.DataFrame(yearly), pd.DataFrame(selections)
    verify(yearly,selections,root,origin)
    summary = summarize(yearly)
    intervals = intervals_2021(root,yearly,origin)
    _, _, dest = calendar_and_paths(root,STATES[0],origin)
    for name, table in (("yearly",yearly),("selection",selections),("summary",summary),
                        ("2021_intervals",intervals)):
        path = dest/f"rolling_margin_{name}.csv"
        temp = path.with_suffix(".csv.tmp")
        table.to_csv(temp,index=False)
        temp.replace(path)
    print(summary[summary.state=="All"].to_string(index=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--first-year",type=int,choices=(2010,2014),default=2010)
    args = parser.parse_args()
    main(args.root.resolve(),args.first_year)
