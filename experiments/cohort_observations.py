"""Account for ACS task eligibility and replay the employment audit on observed-status records."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_temporal_audit import RELSHIPP_TO_RELP, TASKS

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [1, 2]
HISTORICAL = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]


def approval_gaps(scores, groups, outcomes):
    threshold = np.quantile(scores, 0.25)
    approvals = scores >= threshold
    by_group = [approvals[groups == g].mean() for g in (0, 1)]
    favourable = [approvals[(groups == g) & outcomes].mean() for g in (0, 1)]
    return {
        "n": len(scores), "n_group1": int(groups.sum()), "n_favourable": int(outcomes.sum()),
        "threshold": threshold, "approved": int(approvals.sum()),
        "G_pp": 100 * (by_group[1] - by_group[0]),
        "RG_pp": 100 * (favourable[1] - favourable[0]),
    }



def employment_table(root, counts, results):
    ca = counts.loc[counts.state.eq("CA")].set_index("year")
    estimates = results.set_index(["year", "cohort_outcome"])
    lines = [
        r"\begin{table*}[!t]\caption{California Employment Audit by Eligibility for an Employment-Status Record}\label{tab:empstatus}",
        r"\centering\footnotesize\begin{tabular}{r rr rr rrr}\toprule",
        r" & \multicolumn{2}{c}{Black and White respondents} & \multicolumn{2}{c}{Approval gap (pp)} & \multicolumn{3}{c}{Favourable-outcome gap (pp)} \\",
        r"Year & Full cohort & Blank ESR & Full & Recorded & Full & Recorded & Recorded, ESR 1 or 2 \\ \midrule",
    ]
    for year in HISTORICAL:
        full = estimates.loc[(year, "all_records_esr1")]
        observed = estimates.loc[(year, "recorded_status_esr1")]
        expanded = estimates.loc[(year, "recorded_status_esr1_or_2")]
        numbers = [full.G_pp, observed.G_pp, full.RG_pp, observed.RG_pp, expanded.RG_pp]
        row = [str(year), f"{int(full.n):,}", f"{ca.loc[year, 'employment_blank_ESR']:,}"]
        row.extend(f"${number:+.2f}$" for number in numbers)
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule\end{tabular}\par\smallskip",
        r"\begin{minipage}{.96\textwidth}\footnotesize Gaps subtract the White approval rate from the Black approval rate. Each year uses the same XGBoost scores (seed 0) with a 75\% target approval share and no group adjustment. We recompute the score threshold for each cohort. Every blank employment-status record (ESR) in this comparison belongs to a person below age 16. The standard benchmark includes those records with a negative label. Recorded denotes respondents with a reported status. The first two favourable-outcome columns use ESR=1 (civilian employed and at work). The last also includes ESR=2 (civilian with a job but absent from work), holding the restricted cohort and its decisions fixed.\end{minipage}\end{table*}",
    ]
    destination = root / "paper/generated/employment_scope.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf8")

def audit(root):
    cohort_rows, emp_rows = [], []
    states = ("CA", "TX", "NY", "FL")
    years = [2014, *HISTORICAL, 2024]
    expected = {f"{state}_{year}" for state in states for year in years}
    sources = sorted((root / "data/normalized").glob("*.parquet"))
    missing = expected - {source.stem for source in sources}
    if missing:
        raise FileNotFoundError(f"Missing ACS state-year files: {sorted(missing)}")
    run_counts = {}
    for state in states:
        run = root / "results/monitor" / f"{state}_income_race_quantile_d1_s0/run.json"
        for row in json.loads(run.read_text(encoding="utf8"))["years"]:
            run_counts[(state, row["year"])] = row
    for source in (path for path in sources if path.stem in expected):
        state, year_text = source.stem.split("_")
        year = int(year_text)
        frame = pd.read_parquet(source)
        if "RELP" not in frame:
            frame["RELP"] = frame["RELSHIPP"].map(RELSHIPP_TO_RELP)
        income_eligible = (frame.AGEP > 16) & (frame.PINCP > 100) & (frame.WKHP > 0) & (frame.PWGTP >= 1)
        earners = frame.loc[income_eligible]
        race_income = earners.loc[earners.RAC1P.isin(GROUPS)]
        reference = run_counts[(state, year)]
        observed_counts = (len(frame), len(earners), len(race_income), int(race_income.RAC1P.eq(2).sum()))
        reference_counts = tuple(reference[k] for k in ("raw_rows", "filtered_before_group", "analysed", "g1"))
        if observed_counts != reference_counts:
            raise ValueError(f"Cohort counts differ from the model manifest for {state} {year}")
        employment = frame.loc[frame.RAC1P.isin(GROUPS)]
        recorded = employment.ESR.notna()
        score_features = [f for f in TASKS["income"]["features"] if f != "RAC1P"]
        cohort_rows.append({
            "state": state, "year": year, "person_records": len(frame),
            "income_eligible_all_races": len(earners), "income_black_white": len(race_income),
            "income_black": int(race_income.RAC1P.eq(2).sum()), "income_white": int(race_income.RAC1P.eq(1).sum()),
            "income_missing_label_inputs_after_filter": int(earners[["ADJINC", "PINCP"]].isna().any(axis=1).sum()),
            "income_missing_score_features_after_race_filter": int(race_income[score_features].isna().any(axis=1).sum()),
            "employment_black_white": len(employment), "employment_blank_ESR": int((~recorded).sum()),
            "employment_blank_ESR_age_ge_16": int((~recorded & employment.AGEP.ge(16)).sum()),
            "employment_observed_ESR_age_lt_16": int((recorded & employment.AGEP.lt(16)).sum()),
        })
        if state != "CA" or year not in HISTORICAL:
            continue
        score_path = root / "data/scores/audit/CA_employment_race_employed_d1_s0" / f"{year}.npz"
        with np.load(score_path) as data:
            scores, groups = data["s"], data["g"]
            np.testing.assert_array_equal(groups, employment.RAC1P.eq(2).to_numpy())
            np.testing.assert_array_equal(data["y"], employment.ESR.eq(1).to_numpy())
        if not np.array_equal(recorded, employment.AGEP.ge(16)):
            raise ValueError(f"Observed status differs from age eligibility in {state} {year}")
        for cohort, include, labels in [
            ("all_records_esr1", np.ones(len(employment), dtype=bool), employment.ESR.eq(1)),
            ("recorded_status_esr1", recorded, employment.ESR.eq(1)),
            ("recorded_status_esr1_or_2", recorded, employment.ESR.isin([1, 2])),
        ]:
            values = approval_gaps(scores[include], groups[include], labels.to_numpy()[include])
            emp_rows.append({"state": state, "year": year, "cohort_outcome": cohort, **values})
    destination = root / "results/matched"
    destination.mkdir(exist_ok=True, parents=True)
    counts = pd.DataFrame(cohort_rows)
    results = pd.DataFrame(emp_rows)
    counts.to_csv(destination / "acs_cohort_counts.csv", index=False)
    results.to_csv(destination / "employment_status_sensitivity.csv", index=False)
    employment_table(root, counts, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    audit(parser.parse_args().root)
