"""Fetch official one-year ACS PUMS person archives and project raw columns without filtering people."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
FIPS = {"CA": 6, "NY": 36, "FL": 12, "TX": 48}
KEEP = set("STATE ST SERIALNO SPORDER AGEP COW SCHL MAR OCCP POBP RELP RELSHIPP WKHP SEX RAC1P PINCP ADJINC PWGTP DIS ESP CIT MIG MIL ANC NATIVITY DEAR DEYE DREM ESR".split())


def fetch(root, state, year):
    raw_dir = root / "raw"
    slim_dir = root / "normalized"
    for directory in (raw_dir, slim_dir):
        directory.mkdir(parents=True, exist_ok=True)
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{state.lower()}.zip"
    archive = raw_dir / f"{state}_{year}.zip"
    cache = slim_dir / f"{state}_{year}.parquet"
    meta = slim_dir / f"{state}_{year}.json"
    if cache.exists() and meta.exists() and archive.exists():
        return
    if not archive.exists():
        attempt_file = archive.with_suffix(".download")
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=(10, 100)) as response:
                    response.raise_for_status()
                    digest = hashlib.sha256()
                    with attempt_file.open("wb") as writer:
                        for chunk in response.iter_content(1024 * 1024):
                            writer.write(chunk)
                            digest.update(chunk)
                    expected = response.headers.get("content-length")
                    if expected and int(expected) != attempt_file.stat().st_size:
                        raise IOError("truncated download")
                with zipfile.ZipFile(attempt_file) as z:
                    assert any(name.lower().endswith(".csv") for name in z.namelist())
                attempt_file.replace(archive)
                break
            except (OSError, requests.RequestException, zipfile.BadZipFile) as error:
                print(f"{state} {year} attempt={attempt + 1}: {error}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    with archive.open("rb") as source:
        sha = hashlib.file_digest(source, "sha256").hexdigest()
    partial = cache.with_suffix(".partial")
    writer = None
    rows = 0
    try:
        with zipfile.ZipFile(archive) as z:
            csvs = [name for name in z.namelist() if name.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise ValueError(f"Expected one person CSV: {csvs}")
            with z.open(csvs[0]) as source:
                for frame in pd.read_csv(source, skipinitialspace=True, na_values=[" ", "  "], usecols=lambda c: c in KEEP, dtype={col: ("string" if col == "SERIALNO" else "float64") for col in KEEP}, chunksize=65000):
                    state_column = "ST" if "ST" in frame.columns else "STATE"
                    if not frame[state_column].eq(FIPS[state]).all():
                        raise ValueError("Unexpected state in raw file")
                    ar = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(partial, ar.schema, compression="zstd")
                    writer.write_table(ar)
                    rows += len(frame)
    finally:
        if writer:
            writer.close()
    if rows <= 0:
        raise ValueError("Empty census file")
    partial.replace(cache)
    record = {"state":state, "year":year, "source_url":url, "bytes":archive.stat().st_size, "sha256":sha, "zip_member":csvs[0], "raw_person_records":rows, "column_filter_only":True, "columns":pq.read_schema(cache).names}
    meta.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"OK {state} {year}: {rows:,} persons", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=list(FIPS))
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(s, y) for s in args.states for y in args.years]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, args.data_dir, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
