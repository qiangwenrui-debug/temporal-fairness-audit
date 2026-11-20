"""Project origin and race-response flags from the same official person ZIPs as the audit."""
import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATES = ("CA", "TX", "NY", "FL")
YEARS = [*range(2010, 2020), 2021, 2022, 2023, 2024]


def project(job):
    root, state, year = job
    source_dir = root / ("data/early_rolling" if year < 2014 else "data")
    stem = f"{state}_{year}"
    archive = source_dir / "raw" / (stem + ".zip")
    source = source_dir / "normalized" / (stem + ".parquet")
    meta = json.loads(source.with_suffix(".json").read_text(encoding="utf8"))
    with archive.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"Archive differs from the training input: {stem}")
    keys = ["SERIALNO", "SPORDER", "RAC1P"]
    fields = ["HISP", "RACWHT", "RACBLK"]
    with zipfile.ZipFile(archive) as z, z.open(meta["zip_member"]) as csv:
        raw = pd.read_csv(csv, usecols=keys + fields, dtype={"SERIALNO":"string"}, skipinitialspace=True)
    normalized = pd.read_parquet(source, columns=keys)
    for key in keys:
        np.testing.assert_array_equal(raw[key].to_numpy(), normalized[key].to_numpy())
    if raw.duplicated(["SERIALNO","SPORDER"]).any() or not raw.HISP.between(1,24).all():
        raise ValueError(f"Invalid ACS identifiers or origin field: {stem}")
    for field, code in (("RACWHT", 1), ("RACBLK", 2)):
        if not raw[field].isin([0,1]).all() or not raw.loc[raw.RAC1P==code, field].eq(1).all():
            raise ValueError(f"Inconsistent race flags: {stem} {field}")
    output = root / "data/group_reporting" / (stem + ".parquet")
    temp = output.with_suffix(".tmp")
    raw[fields].astype("uint8").to_parquet(temp, compression="zstd",index=False)
    temp.replace(output)
    with output.open("rb") as file:
        projection_hash = hashlib.file_digest(file,"sha256").hexdigest()
    return dict(state=state, year=year, rows=len(raw), zip_sha256=digest,
                projection_sha256=projection_hash, zip_member=meta["zip_member"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    (root/"data/group_reporting").mkdir(parents=True,exist_ok=True)
    jobs = [(root,state,year) for state in STATES for year in YEARS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        manifest = pd.DataFrame(pool.map(project,jobs))
    manifest.to_csv(root/"data/group_reporting/manifest.csv",index=False)
    print(f"Matched {manifest.rows.sum():,} ordered person records in {len(manifest)} archives.")
