"""2024 household-bootstrap intervals with the historical model and policy strength fixed."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from transfer_checks import household_codes

ROOT = Path(__file__).resolve().parents[2]


def frozen_gaps(scores, groups, strengths):
    pooled = np.quantile(scores, 0.25)
    rates = []
    for group in (0, 1):
        subset = scores[groups == group]
        group_cut = np.quantile(subset, 0.25)
        cutoffs = (1 - strengths) * pooled + strengths * group_cut
        rates.append(np.array([(subset >= cut).mean() for cut in cutoffs]))
    return 100 * (rates[1] - rates[0])


def main(root, reps):
    summary = pd.read_csv(root / "results/transfer/transfer_state_summary.csv")
    summary = summary[(summary.scale == "probability") & (summary.budget == "interpolated")]
    if sorted(summary.state) != ["CA", "FL", "NY", "TX"]:
        raise ValueError("Expected one probability-scale frozen-policy comparison in each of the four states")
    rows = []
    for row in summary.itertuples():
        state = row.state
        path = root / "data/scores/audit" / f"{state}_income_race_quantile_d1_s0/2024.npz"
        with np.load(path) as data:
            scores = data["s"].astype(float)
            groups = data["g"]
            households = household_codes(root / f"data/normalized/{state}_2024.parquet", groups, data["y"])
        names = ("pooled", "common")
        strengths = np.array([row.pooled_2024_beta, row.common_2024_beta])
        gaps = frozen_gaps(scores, groups, strengths)
        np.testing.assert_allclose(gaps, [row.pooled_2024_gap_pp, row.common_2024_gap_pp], atol=1e-10)
        n_households = households.max() + 1
        seed = 32024 + {"CA": 6, "TX": 48, "NY": 36, "FL": 12}[state]
        rng = np.random.default_rng(seed)
        draws = np.empty((reps, len(names)))
        people = np.arange(len(scores))
        for i in range(reps):
            sampled = rng.integers(n_households, size=n_households)
            counts = np.bincount(sampled, minlength=n_households)
            indices = np.repeat(people, counts[households])
            draws[i] = frozen_gaps(scores[indices], groups[indices], strengths)
        lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
        for k, policy in enumerate(names):
            rows.append(dict(state=state, year=2024, policy=policy, beta=strengths[k],
                             n_households=n_households, n_people=len(scores), gap_pp=gaps[k],
                             lower_pp=lo[k], upper_pp=hi[k], distance_inside_band_pp=5-abs(gaps[k]),
                             draws=reps, seed=seed))
    out = root / "results/matched"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "frozen_2024_household_intervals.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    main(args.root, args.reps)
