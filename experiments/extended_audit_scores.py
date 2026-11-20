"""Fit the four 2010-origin ACS income score sequences from standard one-year files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import xgboost

from policy_monitor import fitted, panel, save_scores
from policy_monitor_data import YEARS


ROOT = Path(__file__).resolve().parents[2]
STATES = ("CA", "TX", "NY", "FL")
EARLY = list(range(2010, 2014))
LATER = YEARS + [2024]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_state(root, state, threads):
    extended = root / "data/early_rolling"
    early, features, early_meta = panel(extended, state, EARLY, "income", "race", "quantile")
    later, later_features, later_meta = panel(root / "data", state, LATER, "income", "race", "quantile")
    if features != later_features or set(early).intersection(later):
        raise ValueError(f"Incompatible feature order or duplicated survey years for {state}")
    population = {**early, **later}
    metadata = {r["year"]: r for r in early_meta + later_meta}
    years = sorted(population)
    model_id = f"{state}_income_race_quantile_s0"
    task_id = f"{state}_income_race_quantile_d1_s0"
    fits = []
    # Work in an empty directory on every run so a fit from another start date cannot be reused.
    with tempfile.TemporaryDirectory(dir=extended, prefix="fitting_") as directory, threadpool_limits(limits=threads):
        scratch = Path(directory)
        for index, year in enumerate(years[1:], 1):
            past_years = years[:index]
            training = {y: population[y] for y in past_years}
            last = past_years[-1]
            model = fitted(scratch, model_id, training, features, "income", 0, last, threads)
            target = population[year]
            scores = model.predict_proba(target[features])[:, 1]
            if not np.isfinite(scores).all() or not set(target["__g"].unique()) == {0, 1}:
                raise ValueError(f"Invalid scores or group denominator in {state} {year}")
            score_dir = scratch / "scores/audit" / task_id
            save_scores(score_dir, year, scores, target["__g"], target["__y"])
            model_file = Path("models") / model_id / f"{last}.ubj"
            score_file = Path("scores/audit") / task_id / f"{year}.npz"
            n_train = sum(len(frame) for frame in training.values())
            fits.append(dict(state=state, target_year=year, train_years="|".join(map(str,past_years)),
                             training_cutoff=last, eligible_train=n_train, sampled_train=min(400000,n_train),
                             target_n=len(target), target_black=int(target["__g"].sum()),
                             target_raw_sha256=metadata[year]["zip_sha256"],
                             model_sha256=file_hash(scratch/model_file), scores_sha256=file_hash(scratch/score_file),
                             seed=0, features="|".join(features), xgboost=xgboost.__version__,
                             sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
            print(f"{state} {year}: fitted through {last}, scored {len(target):,}",flush=True)
        for kind in ("models","scores"):
            for src in (scratch/kind).rglob("*"):
                if src.is_file():
                    dest = extended / src.relative_to(scratch)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest_tmp = dest.with_suffix(dest.suffix + ".tmp")
                    shutil.copyfile(src, dest_tmp)
                    dest_tmp.replace(dest)
    return fits


def main(root, workers, threads):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(fit_state, root, state, threads) for state in STATES]
        rows = [row for job in jobs for row in job.result()]
    output = root / "results/transfer/extended_fit_manifest.csv"
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix(".csv.tmp")
    with temp.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    main(args.root.resolve(),args.workers,args.threads)
