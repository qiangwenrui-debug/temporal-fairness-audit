"""Expanding-window validation of fixed strengths for the four-state ACS income audit.

Run from any directory. The default verifies the shipped model scores; --refit
also re-trains every state-year model in an empty temporary directory. Both modes
use only packaged parquet files, model/score archives and study results.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_gap_uncertainty import frozen_gaps
from policy_monitor import fitted, panel
from policy_monitor_data import FIPS, YEARS
from run_temporal_audit import BETAS
from transfer_checks import gaps, household_codes, score_curves, smallest


ROOT = Path(__file__).resolve().parents[2]
STATES = ("CA", "TX", "NY", "FL")
Q = .75
DELTA = .05
ARCHIVE_NAMES = {"pooled": "past_pooled_min", "common": "past_common_min",
                 "previous": "carry_previous_min"}


def years_text(years):
    return "|".join(str(year) for year in years)


def calibration_choices(history):
    """Each history entry is from an earlier year's own out-of-time predictions."""
    sizes = np.sum([size for size, counts in history.values()], axis=0)
    approvals = np.sum([counts for size, counts in history.values()], axis=0)
    feasible = [abs(gaps(counts, n)) <= DELTA for n, counts in history.values()]
    return {"pooled": smallest(abs(gaps(approvals, sizes)) <= DELTA),
            "common": smallest(np.all(feasible, axis=0)),
            "previous": smallest(feasible[-1])}


def resampled_intervals(scores, groups, household, strengths, seed, draws):
    n_households = int(household.max()) + 1
    indices = np.arange(len(scores))
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(strengths)))
    for i in range(draws):
        counts = np.bincount(rng.integers(n_households, size=n_households),
                             minlength=n_households)
        people = np.repeat(indices, counts[household])
        estimates[i] = frozen_gaps(scores[people], groups[people], strengths)
    return np.quantile(estimates, [.025, .975], axis=0)


def model_scores(cache_dir, training_dir, refit, state, task_id, population, features,
                 year, training_years, threads):
    last = max(training_years)
    model_id = f"{state}_income_race_quantile_s0"
    if not refit and not (training_dir / "models" / model_id / f"{last}.ubj").is_file():
        raise FileNotFoundError(f"Missing frozen model for {state} through {last}")
    train = {u: population[u] for u in training_years}
    model = fitted(training_dir, model_id, train, features, "income", 0, last, threads)
    cohort = population[year]
    scores = model.predict_proba(cohort[features])[:, 1]
    path = cache_dir / "scores/audit" / task_id / f"{year}.npz"
    with np.load(path) as z:
        np.testing.assert_array_equal(scores, z["s"])
        np.testing.assert_array_equal(cohort["__g"], z["g"])
        np.testing.assert_array_equal(cohort["__y"], z["y"])
    return scores.astype(float), cohort["__g"].to_numpy(), cohort["__y"].to_numpy(), path


def evaluate_state(root, state, years, draws, refit, threads, extra_data):
    task_id = f"{state}_income_race_quantile_d1_s0"
    regular_years = [y for y in years if y >= 2014]
    population, features, metadata = panel(root/"data", state, regular_years, "income", "race", "quantile")
    early_years = [y for y in years if y < 2014]
    if early_years:
        early, early_features, inputs = panel(extra_data, state, early_years, "income", "race", "quantile")
        assert early_features == features
        population.update(early)
        metadata.extend(inputs)
    cache_dir = extra_data if early_years else root/"data"
    raw_hashes = {rec["year"]: rec["zip_sha256"] for rec in metadata}
    history = {}
    records = []
    with tempfile.TemporaryDirectory(prefix="rolling_fits_") as scratch, threadpool_limits(limits=threads):
        training_dir = Path(scratch) if refit else cache_dir
        for year in years[1:]:
            train_years = [u for u in years if u < year]
            # The freeze precedes loading target scores. The first audit seeds the history.
            choices = calibration_choices(history) if history else {}
            score, group, label, score_path = model_scores(
                cache_dir, training_dir, refit, state, task_id, population, features, year, train_years, threads)
            sizes, curves = score_curves(score, group, Q, with_budget=False)
            counts = curves[("probability", "interpolated")]
            gap = gaps(counts, sizes)
            if choices:
                annual = smallest(abs(gap) <= DELTA)
                seed = 32024 + 100*(year-2024) + FIPS[state]
                file_root = extra_data if year < 2014 else root/"data"
                hh = household_codes(file_root / f"normalized/{state}_{year}.parquet", group, label)
                strengths = BETAS[list(choices.values())]
                lower, upper = resampled_intervals(score, group, hh, strengths, seed, draws)
                anchor = np.quantile(score, 1-Q)
                group_anchors = np.array([np.quantile(score[group == g], 1-Q) for g in (0, 1)])
                with score_path.open("rb") as file:
                    score_sha = hashlib.file_digest(file, "sha256").hexdigest()
                available = sum(len(population[t]) for t in train_years)
                common_fields = dict(
                    state=state, year=year, train_years=years_text(train_years),
                    history_years=years_text(history), n_history=len(history),
                    train_through=train_years[-1], latest_audit_year=max(history),
                    train_eligible=available, train_used=min(400000, available),
                    n=len(group), n0=int(sizes[0]), n1=int(sizes[1]), q=Q,
                    raw_gap_pp=100*gap[0], annual_beta=BETAS[annual], annual_gap_pp=100*gap[annual],
                    n_households=int(hh.max())+1, draws=draws, bootstrap_seed=seed,
                    score_sha256=score_sha, target_pums_sha256=raw_hashes[year])
                for j, (policy, k) in enumerate(choices.items()):
                    strength = BETAS[k]
                    cut = (1-strength)*anchor + strength*group_anchors
                    accept = score >= cut[group.astype(int)]
                    direct = np.array([accept[group==g].sum() for g in (0, 1)])
                    np.testing.assert_array_equal(direct, counts[k])
                    differences = counts[k] - counts[0]
                    changed = int(abs(differences).sum())
                    rates = [accept[(group == g) & (label == 1)].mean() for g in (0, 1)]
                    records.append(dict(**common_fields, policy=policy, beta=strength,
                                        tau0=cut[0], tau1=cut[1], approved0=direct[0], approved1=direct[1],
                                        approved=int(accept.sum()), approval_share_pp=100*accept.mean(),
                                        gap_pp=100*gap[k], within=bool(abs(gap[k]) <= DELTA),
                                        distance_inside_band_pp=100*(DELTA-abs(gap[k])),
                                        lower_pp=lower[j], upper_pp=upper[j],
                                        newly_approved=int(np.maximum(differences,0).sum()),
                                        removed=int(np.maximum(-differences,0).sum()),
                                        changed=changed, changed_share_pp=100*changed/len(group),
                                        residual_gap_pp=100*(rates[1]-rates[0])))
            history[year] = (sizes, counts)
            print(f"{state} {year}: {'refit' if refit else 'model'} scores verified; "
                  f"{len(choices)} rolling policies", flush=True)
    return records


def crosscheck_archives(root, data, draws):
    """Reconcile all archived folds, including the former single-year main result."""
    for state, frame in data.groupby("state"):
        old = pd.read_csv(root/f"results/transfer/{state}/through2024/policy_yearly.csv")
        old = old[(old.scale=="probability") & (old.budget=="interpolated")
                  & old.policy.isin(ARCHIVE_NAMES.values())].copy()
        old["policy"] = old.policy.map({v:k for k,v in ARCHIVE_NAMES.items()})
        keys = ["year","policy"]
        cols = ["beta","n","gap_pp","within","approved","newly_approved","removed","changed","changed_share_pp"]
        new = frame[frame.year <= 2024].set_index(keys).sort_index()
        old = old.set_index(keys).sort_index()
        assert old.index.equals(new.index)
        for column in cols:
            np.testing.assert_allclose(new[column], old[column], rtol=0, atol=1e-11)
    if draws == 1000:
        old = pd.read_csv(root/"results/matched/frozen_2024_household_intervals.csv")
        keys = ["state","policy"]
        new = data[(data.year==2024) & data.policy.isin(["pooled","common"])].set_index(keys).sort_index()
        old = old.set_index(keys).sort_index()
        assert old.index.equals(new.index)
        for col in ["beta","gap_pp","lower_pp","upper_pp","n_households"]:
            np.testing.assert_allclose(new[col],old[col],rtol=0,atol=1e-11)


def check_extended_fits(root, data):
    fits = pd.read_csv(root/"results/transfer/extended_fit_manifest.csv",
                       dtype={"train_years":str})
    cols = {"train_years":"train_years","train_through":"training_cutoff",
            "train_used":"sampled_train","train_eligible":"eligible_train",
            "n":"target_n","n1":"target_black","target_pums_sha256":"target_raw_sha256",
            "score_sha256":"scores_sha256"}
    fits = fits.rename(columns={"target_year":"year"}).set_index(["state","year"])
    assert fits.index.is_unique
    folds = data[data.policy=="pooled"].set_index(["state","year"])
    fits = fits.loc[folds.index]
    for field, source in cols.items():
        np.testing.assert_array_equal(folds[field],fits[source])


def main(root, first_year, last_year, draws, refit, workers, threads, extra_data, out):
    if draws < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if first_year > 2014 or last_year < 2024:
        raise ValueError("The study window must include the 2014--2024 comparison")
    years = list(range(first_year,2014)) + YEARS + list(range(2024, last_year+1))
    if first_year < 2014 and extra_data is None:
        raise ValueError("An extended history needs its separate --extra-data-dir with scores and fits")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(evaluate_state,root,state,years,draws,refit,threads,extra_data) for state in STATES]
        for job in jobs:
            rows.extend(job.result())
    data = pd.DataFrame(rows)
    assert len(data) == (len(years)-2)*len(STATES)*3
    assert not data.duplicated(["state","year","policy"]).any()
    assert (data.latest_audit_year == data.train_through).all() and (data.train_through < data.year).all()
    if first_year == 2014:
        crosscheck_archives(root, data, draws)
    if first_year == 2010:
        check_extended_fits(root, data)
    if out is None:
        out = root/"results/transfer" if first_year == 2014 else root/f"results/transfer/from{first_year}"
    out.mkdir(exist_ok=True,parents=True)
    for name, tab in (("rolling_state_year",data[data.policy!="previous"]),
                      ("rolling_previous",data[data.policy=="previous"])):
        target = out/(name+".csv")
        tmp = target.with_suffix(".csv.tmp")
        tab.to_csv(tmp,index=False)
        tmp.replace(target)
    print(data.groupby("policy").within.agg(["sum","size"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--first-year",type=int,default=2014)
    parser.add_argument("--last-year",type=int,default=2024)
    parser.add_argument("--extra-data-dir",type=Path,help="Separate scores and models for the extended origin and pre-2014 normalized person files")
    parser.add_argument("--out",type=Path,help="Output directory (default results/transfer for the 2014 origin; fromYEAR subdirectory otherwise)")
    parser.add_argument("--draws",type=int,default=1000)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--refit",action="store_true",help="Re-train in an empty model directory and verify each person's score")
    args = parser.parse_args()
    main(args.root.resolve(),args.first_year,args.last_year,args.draws,args.refit,args.workers,
         args.threads,args.extra_data_dir,args.out)
