"""Score-scale sensitivity, common feasible strengths and decision turnover.

The snapshot mode reconstructs exact numbers of decisions changed by nested
within-group cutoffs from the archived group sizes, total acceptance and gap.
The score mode reads original-order rows (s, g) from one audit directory.
All shares and differences ending in _pp are in percentage points.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from run_temporal_audit import BETAS


def infer_counts(row, approval, gap):
    n1 = int(row.n1)
    n0 = int(row.n) - n1
    p0 = approval - n1 / row.n * gap
    counts = np.array([n0 * p0, n1 * (p0 + gap)])
    if not np.allclose(counts, np.rint(counts), atol=2e-7, rtol=0):
        raise ValueError("Archived rate or gap did not imply integer approval counts")
    return np.rint(counts).astype(int)


def rank_allocations(initial, sizes, delta):
    """Project the uncalibrated integer group quota onto the feasible interval."""
    budget = int(initial.sum())
    k1 = np.arange(max(0, budget - sizes[0]), min(sizes[1], budget) + 1)
    gap = k1 / sizes[1] - (budget - k1) / sizes[0]
    within = abs(gap) <= delta
    if not within.any():
        raise ValueError("No integer quota satisfies the tolerance and budget")
    min_change = int(k1[within][np.argmin(abs(k1[within] - initial[1]))])
    parity = int(k1[np.argmin(abs(gap))])
    return {k: np.array([budget - v, v]) for k, v in
            (("rank_min_changes", min_change), ("rank_nearest_parity", parity))}


def snapshots(source, out, delta):
    rows = []
    quota_series = {}
    for path in sorted(source.glob("audit_*/yearly.csv")):
        for row in pd.read_csv(path).itertuples():
            if row.model != "xgb":
                continue
            initial = infer_counts(row, row.approval0, row.G0)
            sizes = np.array([row.n - row.n1, row.n1])
            total = initial.sum()
            target = total * sizes[1] / sizes.sum()
            quota1 = np.rint((1 - BETAS) * initial[1] + BETAS * target).astype(int)
            quota = np.column_stack((total - quota1, quota1))
            series = quota_series.setdefault(path.parent.name, {})
            series[int(row.year)] = sizes, {("rank_quota_eta", "keep_realized_budget"): quota}
            policies = {"per_year_min": (infer_counts(row, row.approval_bstar, row.G_bstar), row.beta_star),
                        "pooled_min_retrospective": (infer_counts(row, row.approval_bpool, row.G_bpool), row.beta_pool)}
            for label, counts in rank_allocations(initial, sizes, delta).items():
                policies[label + "_keep_K0"] = (counts, np.nan)
            for policy, (counts, beta) in policies.items():
                gap = counts[1] / sizes[1] - counts[0] / sizes[0]
                change = counts - initial
                rows.append(dict(setting=path.parent.name, year=int(row.year), n=int(row.n),
                                 policy=policy, beta=beta, raw_gap_pp=100 * row.G0,
                                 gap_pp=100 * gap, within=abs(gap) <= delta,
                                 changed_decisions=int(abs(change).sum()),
                                 newly_approved=int(np.maximum(change, 0).sum()),
                                 removed=int(np.maximum(-change, 0).sum()),
                                 changed_share_pp=100 * abs(change).sum() / row.n))
    if not rows:
        raise ValueError(f"No archived audit results under {source}")
    result = pd.DataFrame(rows)
    result.to_csv(out / "archived_yearly.csv", index=False)
    summary = result.groupby(["setting", "policy"], as_index=False).agg(
        years=("year", "size"), within=("within", "sum"), n=("n", "sum"),
        changed_decisions=("changed_decisions", "sum"), max_changed_share_pp=("changed_share_pp", "max"),
        newly_approved=("newly_approved", "sum"), removed=("removed", "sum"))
    summary["pooled_changed_share_pp"] = 100 * summary.changed_decisions / summary.n
    summary.to_csv(out / "archived_summary.csv", index=False)
    yearly, ranges, overall = [], [], []
    for setting, data in quota_series.items():
        a, b, c = evaluate_counts(data, delta, 3)
        yearly.append(a.assign(setting=setting))
        ranges.append(b.assign(setting=setting))
        overall.append(c.assign(setting=setting))
    pd.concat(yearly, ignore_index=True).rename(columns={"beta": "eta"}).to_csv(out / "archived_quota_yearly.csv", index=False)
    pd.concat(ranges, ignore_index=True).rename(columns={"beta_min": "eta_min", "beta_pool_retrospective": "eta_pool_retrospective"}).to_csv(out / "archived_quota_ranges.csv", index=False)
    pd.concat(overall, ignore_index=True).to_csv(out / "archived_quota_summary.csv", index=False)


def smallest(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("No feasible strength on the specified grid")
    return int(indices[0])


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return "|".join(f"{BETAS[i]:.3f}..{BETAS[j]:.3f}" for i, j in
                    zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def exact_budget(sorted_scores, sorted_ids, group_offsets, budget):
    """Number approved by group among the exact top k shifted scores.

    Binary partition of two sorted arrays, one shifted relative to the other.
    Among equal margins the earlier original row is approved first, independent
    of its group. Stable within-group ranking permits exact turnover from counts.
    """
    s0, s1 = sorted_scores
    ids0, ids1 = sorted_ids
    rejected = len(s0) + len(s1) - budget
    left = np.full(len(group_offsets), max(0, rejected - len(s0)), int)
    right = np.full(len(group_offsets), min(len(s1), rejected), int)
    while np.any(left < right):
        selected1 = (left + right) // 2
        selected0 = rejected - selected1
        i1 = np.minimum(selected1, len(s1) - 1)
        i0 = np.maximum(selected0 - 1, 0)
        value1 = s1[i1] - group_offsets
        value0 = s0[i0]
        less = (value1 < value0) | ((value1 == value0) & (ids1[i1] > ids0[i0]))
        need_more = (selected0 > 0) & (selected1 < len(s1)) & less & (left < right)
        left = np.where(need_more, selected1 + 1, left)
        right = np.where(need_more, right, selected1)
    return np.column_stack((len(s0) - (rejected - left), len(s1) - left))


def score_curves(scores, groups, q, with_budget):
    scores = np.asarray(scores, dtype=float)
    if not (np.isfinite(scores).all() and scores.min() > 0 and scores.max() < 1):
        raise ValueError("Strict logit comparison requires scores in (0, 1)")
    if not np.array_equal(np.unique(groups), [0, 1]):
        raise ValueError("Both binary groups must be present")
    order = []
    for group in (0, 1):
        ids = np.flatnonzero(groups == group)
        sorted_ids = np.lexsort((-ids, scores[ids])) if with_budget else np.argsort(scores[ids])
        order.append(ids[sorted_ids])
    n_by_group = np.array([len(order[0]), len(order[1])])
    anchor = np.array([np.quantile(scores, 1 - q)] +
                      [np.quantile(scores[order[g]], 1 - q) for g in (0, 1)])
    curves = {}
    for name, transform in (("probability", np.asarray), ("logit", logit)):
        global_cut, cut0, cut1 = transform(anchor)
        sorted_score = [transform(scores[order[g]]) for g in (0, 1)]
        cutoffs = np.array([(1 - BETAS) * global_cut + BETAS * v for v in (cut0, cut1)])
        counts = np.column_stack([n_by_group[g] - np.searchsorted(sorted_score[g], cutoffs[g])
                                  for g in (0, 1)])
        curves[(name, "interpolated")] = counts
        if with_budget:
            curves[(name, "exact_budget")] = exact_budget(sorted_score, order, cutoffs[1] - cutoffs[0],
                                                        int(round(q * len(scores))))
    return n_by_group, curves


def gaps(counts, sizes):
    return counts[:, 1] / sizes[1] - counts[:, 0] / sizes[0]


def evaluate(score_files, out, q, delta, early_n):
    data = {}
    for year, path in score_files.items():
        with np.load(path) as z:
            data[year] = score_curves(z["s"], z["g"], q, with_budget=True)
    result, feasibility, summary = evaluate_counts(data, delta, early_n)
    result.to_csv(out / "policy_yearly.csv", index=False)
    feasibility.to_csv(out / "feasible_grid.csv", index=False)
    summary.to_csv(out / "policy_summary.csv", index=False)


def evaluate_counts(data, delta, early_n):
    years = sorted(data)
    if len(years) <= early_n or early_n < 1:
        raise ValueError("An early calibration window and at least one later year are required")
    records, feasibility = [], []
    for scale, rule in data[years[0]][1]:
        counts = {t: data[t][1][(scale, rule)] for t in years}
        sizes = {t: data[t][0] for t in years}
        curves = np.stack([gaps(counts[t], sizes[t]) for t in years])
        feasible = abs(curves) <= delta
        all_ok = feasible.all(axis=0)
        early = years[:early_n]
        per_year = [smallest(a) for a in feasible]
        summed_counts = sum(counts.values())
        pooled = smallest(abs(gaps(summed_counts, sum(sizes.values()))) <= delta)
        all_intersection = smallest(all_ok)
        early_intersection = smallest(feasible[:early_n].all(axis=0))
        early_margin = smallest((abs(curves[:early_n]) <= delta / 2).all(axis=0))
        for i, year in enumerate(years):
            feasibility.append(dict(scale=scale, budget=rule, year=year,
                                    feasible_grid_intervals=intervals(feasible[i]),
                                    joint_ranges_all_years=intervals(all_ok),
                                    joint_ranges_early=intervals(feasible[:early_n].all(axis=0)),
                                    first_years=",".join(map(str, early)), beta_min=BETAS[per_year[i]],
                                    beta_pool_retrospective=BETAS[pooled]))
            policy = {"uncalibrated": 0, "per_year_min": per_year[i],
                      "pooled_min_retrospective": pooled, "common_min_retrospective": all_intersection,
                      "parity_endpoint": len(BETAS) - 1}
            if i:
                policy["carry_previous_min"] = per_year[i - 1]
                policy["first_year_min_frozen"] = per_year[0]
                previous = years[:i]
                historical_pool = gaps(sum(counts[t] for t in previous), sum(sizes[t] for t in previous))
                policy["past_pooled_min"] = smallest(abs(historical_pool) <= delta)
                policy["past_common_min"] = smallest(feasible[:i].all(axis=0))
                policy["past_half_tolerance_common"] = smallest((abs(curves[:i]) <= delta / 2).all(axis=0))
            if year > early[-1]:
                policy["early_common_frozen"] = early_intersection
                policy["early_half_tolerance_frozen"] = early_margin
            for label, k in policy.items():
                selected = counts[year][k]
                diff = selected - counts[year][0]
                nr = int(sizes[year].sum())
                records.append(dict(scale=scale, budget=rule, year=year, after_early=year > early[-1],
                                    policy=label, beta=BETAS[k], n=nr, gap_pp=100 * curves[i, k],
                                    within=bool(feasible[i, k]), approved=int(selected.sum()),
                                    changed=int(abs(diff).sum()), newly_approved=int(np.maximum(diff, 0).sum()),
                                    removed=int(np.maximum(-diff, 0).sum()), changed_share_pp=100 * abs(diff).sum() / nr))
            if rule == "exact_budget":
                initial = counts[year][0]
                for label, selected in rank_allocations(initial, sizes[year], delta).items():
                    diff = selected - initial
                    gap = selected[1] / sizes[year][1] - selected[0] / sizes[year][0]
                    records.append(dict(scale=scale, budget=rule, year=year, after_early=year > early[-1],
                                        policy=label, beta=np.nan, n=nr, gap_pp=100 * gap, within=abs(gap) <= delta,
                                        approved=int(selected.sum()), changed=int(abs(diff).sum()),
                                        newly_approved=int(np.maximum(diff, 0).sum()), removed=int(np.maximum(-diff, 0).sum()),
                                        changed_share_pp=100 * abs(diff).sum() / nr))
    result = pd.DataFrame(records)
    summaries = []
    for subset, frame in (("all_available", result), ("shared_after_early", result[result.after_early])):
        summary = frame.groupby(["scale", "budget", "policy"], as_index=False).agg(
            years=("year", "size"), within=("within", "sum"), mean_abs_gap_pp=("gap_pp", lambda x: x.abs().mean()),
            maximum_abs_gap_pp=("gap_pp", lambda x: x.abs().max()), changed=("changed", "sum"),
            max_changed_share_pp=("changed_share_pp", "max"), n=("n", "sum"))
        summary["pooled_changed_share_pp"] = 100 * summary.changed / summary.n
        summaries.append(summary.assign(evaluation=subset))
    return result, pd.DataFrame(feasibility), pd.concat(summaries, ignore_index=True)


def scale_contrasts(s, g, q, delta):
    sizes, curves = score_curves(s, g, q, with_budget=False)
    metrics = []
    for scale in ("probability", "logit"):
        counts = curves[(scale, "interpolated")]
        k = smallest(abs(gaps(counts, sizes)) <= delta)
        metrics.append((BETAS[k], counts[k], counts[0]))
    prob, logits = metrics
    if not np.array_equal(prob[2], logits[2]):
        raise AssertionError("Score transformations must preserve the uncalibrated decisions")
    return np.array([logits[0] - prob[0], 100 * abs(logits[1] - prob[1]).sum() / len(s)])


def household_codes(parquet, expected_group, expected_label):
    cols = ["SERIALNO", "AGEP", "PINCP", "WKHP", "PWGTP", "ADJINC", "RAC1P"]
    raw = pd.read_parquet(parquet, columns=cols)
    earners = raw[(raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)]
    income = earners.PINCP * earners.ADJINC / 1e6
    y_all = income > income.quantile(.6)
    keep = earners.RAC1P.isin([1, 2])
    g = (earners.RAC1P[keep] == 2).to_numpy()
    if not np.array_equal(g, expected_group) or not np.array_equal(y_all[keep].to_numpy(), expected_label):
        raise ValueError("Household identifiers did not reproduce cached groups and outcomes")
    ids = earners.SERIALNO[keep]
    if ids.isna().any():
        raise ValueError("Missing household serial numbers")
    return pd.factorize(ids, sort=False)[0]


def paired_bootstrap(score_files, out, q, delta, reps, seed, household_dir, state):
    rows = []
    for year, path in sorted(score_files.items()):
        with np.load(path) as z:
            s, g, y = z["s"], z["g"], z["y"]
        rng = np.random.default_rng([seed, year])
        clusters = household_codes(household_dir / f"{state}_{year}.parquet", g, y) if household_dir else None
        n_clusters = int(clusters.max() + 1) if clusters is not None else len(s)
        estimates = scale_contrasts(s, g, q, delta)
        draws = []
        for _ in range(reps):
            selected = rng.integers(0, n_clusters, size=n_clusters)
            if clusters is None:
                i = selected
            else:
                freq = np.bincount(selected, minlength=n_clusters)
                i = np.repeat(np.arange(len(s)), freq[clusters])
            draws.append(scale_contrasts(s[i], g[i], q, delta))
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        for metric, value, lo, hi in zip(("beta_logit_minus_probability", "changed_between_scales_pp"),
                                         estimates, lower, upper):
            rows.append(dict(year=year, metric=metric, estimate=value, lower95=lo, upper95=hi,
                             replicates=reps, seed=seed, resampling_unit="SERIALNO" if household_dir else "person",
                             n_units=n_clusters, n_persons=len(g)))
    pd.DataFrame(rows).to_csv(out / "paired_scale_bootstrap.csv", index=False)


def compile_tables(root, out):
    rows = []
    for state in ("CA", "TX", "NY", "FL"):
        path = root / state
        ranges = pd.read_csv(path / "historical/feasible_grid.csv")
        history = pd.read_csv(path / "historical/policy_summary.csv")
        extended = pd.read_csv(path / "through2024/policy_yearly.csv")
        for (scale, budget), group in ranges.groupby(["scale", "budget"], sort=False):
            hist = history[(history.scale == scale) & (history.budget == budget) &
                           (history.evaluation == "all_available")].set_index("policy")
            held = extended[(extended.scale == scale) & (extended.budget == budget) &
                            (extended.year == 2024)].set_index("policy")
            row = dict(state=state, scale=scale, budget=budget, historical_years=len(group),
                       historical_feasible_strengths=group.joint_ranges_all_years.iloc[0])
            for short, old, new in (("pooled", "pooled_min_retrospective", "past_pooled_min"),
                                    ("common", "common_min_retrospective", "past_common_min"),
                                    ("adaptive", "per_year_min", "per_year_min"),
                                    ("parity", "parity_endpoint", "parity_endpoint")):
                row[short + "_historical_within"] = hist.loc[old, "within"]
                row[short + "_historical_changed_pp"] = hist.loc[old, "pooled_changed_share_pp"]
                for var in ("beta", "gap_pp", "changed_share_pp", "within"):
                    row[short + "_2024_" + var] = held.loc[new, var]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "transfer_state_summary.csv", index=False)
    seed_rows = []
    for seed in range(5):
        folder = root / "CA" / (f"seed{seed}" if seed else "historical")
        config = pd.read_csv(folder / "feasible_grid.csv")
        results = pd.read_csv(folder / "policy_summary.csv")
        for (scale, budget), group in config.groupby(["scale", "budget"], sort=False):
            r = results[(results.scale == scale) & (results.budget == budget) &
                        (results.evaluation == "all_available")].set_index("policy")
            seed_rows.append(dict(seed=seed, scale=scale, budget=budget, common_range=group.joint_ranges_all_years.iloc[0],
                                  pooled_beta=group.beta_pool_retrospective.iloc[0],
                                  pooled_within=r.loc["pooled_min_retrospective", "within"],
                                  common_changed_pp=r.loc["common_min_retrospective", "pooled_changed_share_pp"],
                                  adaptive_changed_pp=r.loc["per_year_min", "pooled_changed_share_pp"]))
    pd.DataFrame(seed_rows).to_csv(out / "transfer_seed_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-dir", type=Path)
    source.add_argument("--scores", type=Path)
    source.add_argument("--combine-states", type=Path, help="Compile four-state and five-seed runs into two tables")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    parser.add_argument("--early-n", type=int, default=3)
    parser.add_argument("--q", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=9421)
    parser.add_argument("--acs-household-dir", type=Path, help="Normalized ACS parquet directory; income/race/quantile only")
    parser.add_argument("--state", default="CA", help="Used only to locate household identifiers")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.archive_dir:
        snapshots(args.archive_dir, args.out, args.delta)
        return
    if args.combine_states:
        compile_tables(args.combine_states, args.out)
        return
    files = {year: args.scores / f"{year}.npz" for year in args.years}
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Requested complete series is missing {missing}")
    evaluate(files, args.out, args.q, args.delta, args.early_n)
    if args.bootstrap:
        paired_bootstrap(files, args.out, args.q, args.delta, args.bootstrap, args.bootstrap_seed,
                         args.acs_household_dir, args.state)


if __name__ == "__main__":
    main()
