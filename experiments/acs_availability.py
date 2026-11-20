"""Inventory official state ACS one-year person ZIPs and check the archived study files."""
import argparse
import csv
import hashlib
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
FTP = "https://www2.census.gov/programs-surveys/acs/"
STANDARD = FTP + "data/pums/"
EXPERIMENTAL = FTP + "experimental/2020/data/pums/1-Year/"
STATES = ("CA", "TX", "NY", "FL")
INCOME_COLUMNS = set("AGEP COW SCHL MAR OCCP POBP WKHP SEX RAC1P PINCP ADJINC PWGTP SERIALNO".split())


def request(url, *, directory, full_digest=False):
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 40)) as response:
                if response.status_code == 404:
                    return 404, "", None
                response.raise_for_status()
                if directory:
                    return response.status_code, response.text, None
                chunks = response.iter_content(1024 * 1024 if full_digest else 4)
                first = next(chunks, b"")
                if first[:4] != b"PK\x03\x04":
                    raise ValueError(f"Expected a ZIP local file header at {url}")
                length = response.headers.get("Content-Length")
                signature = "zip"
                if full_digest:
                    sha = hashlib.sha256(first)
                    for chunk in chunks:
                        sha.update(chunk)
                    signature = sha.hexdigest()
                return response.status_code, signature, int(length) if length else None
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def standard_directory(year):
    # Before annual and multiyear products shared a release year, the files sat
    # directly below the year directory.
    suffix = "/" if year < 2007 else "/1-Year/"
    return f"{STANDARD}{year}{suffix}"


def study_file(year, state):
    data_root = ROOT / "data"
    if year < 2014:
        data_root = data_root / "early_rolling"
    base = f"{state}_{year}"
    meta = json.loads((data_root / "normalized" / (base + ".json")).read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(data_root / "normalized" / (base + ".parquet"))
    fields = set(parquet.schema_arrow.names)
    if not INCOME_COLUMNS <= fields or not {"RELP", "RELSHIPP"} & fields:
        raise ValueError(f"{base}: missing income inputs")
    if parquet.metadata.num_rows != meta["raw_person_records"]:
        raise ValueError(f"{base}: inconsistent raw and normalized person counts")
    raw = data_root / "raw" / (base + ".zip")
    if raw.exists():
        with raw.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with zipfile.ZipFile(raw) as archive:
            if meta["zip_member"] not in archive.namelist():
                raise ValueError(f"{base}: missing person file in archive")
        if digest != meta["sha256"] or raw.stat().st_size != meta["bytes"]:
            raise ValueError(f"{base}: raw data differ from acquisition manifest")
    # Count the same income-task population used by the scoring pipeline.
    relation = "RELP" if "RELP" in fields else "RELSHIPP"
    columns = sorted((INCOME_COLUMNS - {"SERIALNO"}) | {relation})
    frame = parquet.read(columns=columns).to_pandas()
    eligible = frame.loc[(frame.AGEP > 16) & (frame.PINCP > 100) & (frame.WKHP > 0) & (frame.PWGTP >= 1)]
    compared = eligible.loc[eligible.RAC1P.isin([1, 2])]
    model_inputs = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", relation, "WKHP", "SEX"]
    counts = dict(eligible_people=len(eligible), black_earners=int(compared.RAC1P.eq(2).sum()),
                  white_earners=int(compared.RAC1P.eq(1).sum()),
                  missing_income_inputs=int(eligible[["PINCP", "ADJINC"]].isna().any(axis=1).sum()),
                  missing_score_inputs=int(compared[model_inputs].isna().any(axis=1).sum()))
    return meta, raw.exists(), counts


def inspect(job):
    year, product, state, url, listing, full_digest = job
    filename = f"csv_p{state.lower()}.zip"
    archive_url = url + filename
    status, signature, byte_size = request(archive_url, directory=False, full_digest=full_digest)
    listed = filename in listing
    if listed != (status == 200):
        raise ValueError(f"Disagreement between archive and directory: {archive_url}")
    in_study = listed and product == "standard" and 2010 <= year <= 2024
    row = dict(year=year, state=state, product=product, directory_url=url,
               person_archive_url=archive_url, listed=int(listed), http_status=status,
               zip_signature_verified=int(status == 200), http_bytes=byte_size,
               included_2010_start=int(in_study), included_2014_start=int(in_study and year >= 2014),
               cached_person_records=None, cached_zip_bytes=None, cached_zip_sha256="",
               recomputed_local_zip_sha256=None, independently_downloaded_sha256="",
               eligible_people=None, black_earners=None, white_earners=None,
               missing_income_inputs=None, missing_score_inputs=None)
    if in_study:
        meta, verified, counts = study_file(year, state)
        if meta["source_url"] != archive_url or (byte_size is not None and byte_size != meta["bytes"]):
            raise ValueError(f"{year} {state}: acquisition URL or file length has changed")
        if full_digest and signature != meta["sha256"]:
            raise ValueError(f"{year} {state}: newly downloaded archive differs from the recorded digest")
        row.update(cached_person_records=meta["raw_person_records"], cached_zip_bytes=meta["bytes"],
                   cached_zip_sha256=meta["sha256"], recomputed_local_zip_sha256=int(verified),
                   independently_downloaded_sha256=signature if full_digest else "", **counts)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="CSV for a dated, live audit")
    parser.add_argument("--through", default=2025, type=int, help="Last survey year to probe")
    parser.add_argument("--verify-early-downloads", action="store_true",
                        help="Stream 2010–2013 ZIPs in full and compare to the acquisition digests")
    args = parser.parse_args()
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    directories = [(y, "standard", standard_directory(y)) for y in range(2005, args.through + 1)]
    if args.through >= 2020:
        directories.append((2020, "experimental", EXPERIMENTAL))
    jobs = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for (year, product, url), (_, html, _) in zip(directories, pool.map(
                lambda row: request(row[2], directory=True), directories)):
            listing = set(re.findall(r'href="([^"]+)"', html, flags=re.I))
            full_digest = args.verify_early_downloads and 2010 <= year <= 2013 and product == "standard"
            jobs.extend((year, product, state, url, listing, full_digest) for state in STATES)
        rows = list(pool.map(inspect, jobs))
    rows.sort(key=lambda row: (row["year"], row["product"], STATES.index(row["state"])))
    for row in rows:
        row["checked_at_utc"] = checked_at
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Checked {len(rows)} state/product/year entries; "
          f"{sum(row['included_2010_start'] for row in rows)} in the extended study.")


if __name__ == "__main__":
    main()
