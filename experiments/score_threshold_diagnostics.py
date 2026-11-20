"""Recover score distributions, threshold ranks and decision counts in the 2010-start audit."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT / "results/transfer/from2010"


def score_path(state, year):
    folder = ROOT / "data/early_rolling/scores/audit" / f"{state}_income_race_quantile_d1_s0"
    return folder / f"{year}.npz"


def cdf_counts(sorted_scores, cutoffs):
    below = np.searchsorted(sorted_scores, cutoffs, side="left")
    equal = np.searchsorted(sorted_scores, cutoffs, side="right") - below
    return below, equal


def build():
    source = pd.concat([pd.read_csv(TRANSFER/name, float_precision="round_trip")
                        for name in ("rolling_state_year.csv", "rolling_previous.csv")])
    if source.duplicated(["state", "year", "policy"]).any():
        raise ValueError("A policy is duplicated in a cohort.")
    profiles, thresholds, cdfs = [], [], []
    target_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
    for state in ("CA", "TX", "NY", "FL"):
        for year in [2011, *target_years]:
            with np.load(score_path(state, year)) as cache:
                scores, groups = cache["s"].astype(float), cache["g"]
            ordered = [np.sort(scores[groups == group]) for group in (0, 1)]
            anchor = np.quantile(scores, .25)
            group_anchor = np.array([np.quantile(part, .25) for part in ordered])
            rules = source[(source.state == state) & (source.year == year)]
            if year in target_years and set(rules.policy) != {"pooled", "previous", "common"}:
                raise ValueError(f"Incomplete policy set in {state} {year}.")
            cuts = [[], []]
            for row in rules.itertuples():
                tau = (1-row.beta)*anchor + row.beta*group_anchor
                annual_tau = (1-row.annual_beta)*anchor + row.annual_beta*group_anchor
                np.testing.assert_allclose(tau, [row.tau0, row.tau1], atol=1e-14, rtol=0)
                for group, part in enumerate(ordered):
                    cut = tau[group]
                    n = len(part)
                    below, at = cdf_counts(part, cut)
                    below_annual, _ = cdf_counts(part, annual_tau[group])
                    approved = int(n-below)
                    if approved != getattr(row, f"approved{group}") or n != getattr(row, f"n{group}"):
                        raise ValueError(f"Threshold counts changed for {state} {year}.")
                    thresholds.append(dict(state=state, year=year, group=["White","Black"][group],
                        policy=row.policy, n=n, n_approved=approved, n_below=int(below), n_at_cut=int(at),
                        rejection_pct=100*below/n, approval_pct=100*approved/n, pooled_q25=anchor,
                        group_q25=group_anchor[group], beta=row.beta, threshold=cut, gap_pp=row.gap_pp,
                        target_year_min_beta=row.annual_beta, target_year_min_threshold=annual_tau[group],
                        target_year_min_approved=int(n-below_annual),
                        approval_count_change_to_target_min=int(below-below_annual)))
                    cuts[group].append(cut)
            for group, part in enumerate(ordered):
                key = dict(state=state, year=year, group=["White", "Black"][group])
                quantiles = np.quantile(part, np.arange(101)/100)
                profiles.append(dict(**key, n=len(part), group_share_pct=100*len(part)/len(scores),
                                     pooled_q25=anchor, **{f"p{i:02}":val for i,val in enumerate(quantiles)}))
                grid = np.unique(np.r_[np.linspace(0,1,201), cuts[group], group_anchor[group], anchor])
                below, at = cdf_counts(part, grid)
                cdfs.extend(dict(**key, score=x, n=len(part), n_below=int(l), n_equal=int(e),
                                 pct_below=100*l/len(part)) for x,l,e in zip(grid,below,at))
    out = TRANSFER / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("group_score_quantiles",profiles), ("group_thresholds",thresholds),
                          ("group_score_ecdf",cdfs)):
        pd.DataFrame(records).to_csv(out/(name+".csv"),index=False)
    assert len(thresholds) == 4*12*3*2 and len(profiles) == 4*13*2


if __name__ == "__main__":
    build()
