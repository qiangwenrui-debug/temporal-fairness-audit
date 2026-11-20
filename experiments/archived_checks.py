"""Matched-window credit analysis and replicate-paired simulation comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input_results"
OUT = ROOT / "results" / "matched"


def synthetic():
    runs = pd.read_csv(INPUT / "revision_synthetic" / "part1.csv")
    keys = ["scenario", "rep"]
    runs = runs.sort_values(keys + ["t"]).rename(columns={"E0": "E0prime"})
    runs["E0"] = runs.groupby(keys)["RG"].shift()
    common = runs.dropna(subset=["E0"]).copy()
    assert (common.groupby(keys).size() == 6).all()
    estimators = ["E0", "E0prime", "E1", "E2", "E3u", "E3", "E3_oracle"]
    for name in estimators:
        common[name] = 100 * (common[name] - common.RG).abs()
    per_rep = common.groupby(keys)[estimators].mean().reset_index()
    per_rep.to_csv(OUT / "simulation_replicate_mae.csv", index=False)
    rng = np.random.default_rng(4821)
    rows = []
    for scenario, d in per_rep.groupby("scenario"):
        assert len(d) == 20
        pick = rng.integers(len(d), size=(10000, len(d)))
        row = {"scenario": scenario}
        row.update(d[estimators].mean().to_dict())
        for base in ["E2", "E3u"]:
            delta = (d.E3 - d[base]).to_numpy()
            row["E3_minus_" + base] = delta.mean()
            ci = np.quantile(delta[pick].mean(axis=1), [.025, .975])
            row[base + "_lo"], row[base + "_hi"] = ci
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "simulation_paired.csv", index=False)


def credit():
    d = pd.read_csv(INPUT / "csmar_timing_check" / "timing_check_yearly.csv")
    d = d[d.year.between(2014, 2021)].copy()
    assert (d.groupby(["design", "model"]).size() == 8).all()
    rows = []
    for (design, model), frame in d.groupby(["design", "model"]):
        gap = 100 * frame.gap
        rows.append(dict(design=design, model=model, years=len(frame),
                         mean_annual_pp=gap.mean(), mean_absolute_annual_pp=gap.abs().mean(),
                         minimum_pp=gap.min(), maximum_pp=gap.max(),
                         outside=int((gap.abs() > 5).sum())))
    pd.DataFrame(rows).to_csv(OUT / "credit_common_calendar.csv", index=False)


def decomposition():
    d = pd.read_csv(INPUT / "results_income_race_CA_d1_quantile" / "yearly.csv")
    d = d[(d.model == "xgb") & np.isclose(d.q, .75)].copy()
    for g in (0, 1):
        d[f"a{g}1"] = 1 - d[f"FRR{g}"]
        d[f"a{g}0"] = d[f"FAR{g}"]
    rc = (1-d.pi1)*(d.a10-d.a11) - (1-d.pi0)*(d.a00-d.a01)
    np.testing.assert_allclose(d.RC, rc, atol=1e-10)
    np.testing.assert_allclose(d.G, d.RG + rc, atol=1e-10)
    cols = ["year", "n", "n1", "pi1", "pi0", "a11", "a01", "a10", "a00", "G", "RG", "RC"]
    d[cols].to_csv(OUT / "california_components.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic()
    credit()
    decomposition()
