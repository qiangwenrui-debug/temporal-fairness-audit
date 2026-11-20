"""Pointwise household intervals for the three fixed margins and the margin selected from past years."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from model_cohort_replay import BootstrapScores, sample_weights
from transfer_checks import household_codes


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/transfer/from2010"
POLICIES = ["fixed_0", "fixed_0_5", "fixed_1", "rolling"]


def state_intervals(state):
    policies = pd.read_csv(RESULTS / "rolling_margin_yearly.csv", float_precision="round_trip")
    baseline = pd.read_csv(RESULTS / "rolling_state_year.csv", float_precision="round_trip")
    baseline = baseline[(baseline.state == state) & (baseline.policy == "common")].set_index("year")
    rows = []
    for year, fold in policies[policies.state == state].groupby("year"):
        fold = fold.set_index("policy").loc[POLICIES]
        reference = baseline.loc[year]
        extended = ROOT / "data/early_rolling"
        data = extended if year < 2014 else ROOT / "data"
        with np.load(extended / f"scores/audit/{state}_income_race_quantile_d1_s0/{year}.npz") as z:
            scores = BootstrapScores(z["s"].astype(float), z["g"])
            households = household_codes(data / f"normalized/{state}_{year}.parquet", z["g"], z["y"])
        strengths = fold.beta.to_numpy()
        _, gaps, _ = scores.thresholds_and_gap(np.ones(len(households), dtype=int), strengths)
        np.testing.assert_allclose(gaps, fold.gap_pp, rtol=0, atol=1e-11)
        draws, seed = int(reference.draws), int(reference.bootstrap_seed)
        rng = np.random.default_rng(seed)
        sampled = np.empty((draws,len(POLICIES)))
        for b in range(draws):
            _, sampled[b], _ = scores.thresholds_and_gap(sample_weights(households,rng), strengths)
        lower, upper = np.quantile(sampled, [.025,.975], axis=0)
        np.testing.assert_allclose([lower[0],upper[0]], [reference.lower_pp,reference.upper_pp], rtol=0, atol=1e-10)
        selected_idx = np.flatnonzero(fold.margin_pp.iloc[:3].to_numpy() == fold.margin_pp.iloc[-1]).item()
        np.testing.assert_array_equal(sampled[:,selected_idx],sampled[:,-1])
        for j, policy in enumerate(POLICIES):
            rows.append(dict(state=state,year=year,policy=policy,margin_pp=fold.margin_pp.iloc[j],
                             gap_pp=gaps[j],lower_pp=lower[j],upper_pp=upper[j],draws=draws,seed=seed))
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = [r for group in pool.map(state_intervals, ("CA","TX","NY","FL")) for r in group]
    frame = pd.DataFrame(records)
    if len(frame)!=192 or frame.duplicated(["state","year","policy"]).any():
        raise ValueError("Incomplete margin intervals for the primary panel")
    out = RESULTS / "rolling_margin_intervals.csv"
    temp = out.with_suffix(".csv.tmp")
    frame.to_csv(temp,index=False)
    temp.replace(out)
