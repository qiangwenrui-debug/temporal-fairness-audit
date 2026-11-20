"""Replay tighter all-history constraints, retaining the five-point test criterion."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extended_audit_scores import EARLY, LATER, ROOT, STATES
from run_temporal_audit import BETAS
from transfer_checks import gaps, score_curves, smallest


HISTORICAL_TOLERANCES = (.05,.045,.04)


def main(root):
    target = root/"results/transfer/from2010"
    observed = pd.read_csv(target/"rolling_state_year.csv",float_precision="round_trip")
    observed = observed[observed.policy=="common"].set_index(["state","year"])
    result = []
    for state in STATES:
        history = {}
        directory = root/"data/early_rolling/scores/audit"/f"{state}_income_race_quantile_d1_s0"
        for year in (EARLY+LATER)[1:]:
            if history:
                past = np.stack(list(history.values()))
                worst = np.max(np.abs(past),axis=0)
                frozen = [smallest(worst<=tol) for tol in HISTORICAL_TOLERANCES]
            with np.load(directory/f"{year}.npz") as arrays:
                score = arrays["s"].astype(float)
                group = arrays["g"]
            sizes, curves = score_curves(score,group,.75,with_budget=False)
            counts = curves[("probability","interpolated")]
            annual_gap = gaps(counts,sizes)
            if history:
                baseline_idx = frozen[0]
                base = counts[baseline_idx]
                actual = observed.loc[(state,year)]
                np.testing.assert_array_equal(base,[actual.approved0,actual.approved1])
                assert abs(BETAS[baseline_idx]-actual.beta)<1e-12
                pooled = np.quantile(score,.25)
                group_q = np.array([np.quantile(score[group==g],.25) for g in (0,1)])
                original_cut = (1-BETAS[baseline_idx])*pooled + BETAS[baseline_idx]*group_q
                base_decision = score>=original_cut[group]
                for tol,k in zip(HISTORICAL_TOLERANCES,frozen):
                    cut = (1-BETAS[k])*pooled + BETAS[k]*group_q
                    decision = score>=cut[group]
                    gained = int(np.count_nonzero(decision & ~base_decision))
                    removed = int(np.count_nonzero(base_decision & ~decision))
                    change_raw = int(np.count_nonzero(decision != (score>=pooled)))
                    n = len(score)
                    assert gained+removed == abs(counts[k]-base).sum()
                    assert change_raw == abs(counts[k]-counts[0]).sum()
                    inside = abs(annual_gap[k]) <= .05
                    old_inside = bool(actual.within)
                    result.append(dict(state=state,year=year,history_years="|".join(map(str,history)),
                                       historical_tolerance_pp=100*tol,test_tolerance_pp=5.,
                                       beta=BETAS[k],historical_worst_absolute_gap_pp=100*worst[k],
                                       n=n,n0=int(sizes[0]),n1=int(sizes[1]),approved0=int(counts[k,0]),
                                       approved1=int(counts[k,1]),tau0=cut[0],tau1=cut[1],
                                       gap_pp=100*annual_gap[k],within=inside,baseline_within=old_inside,
                                       newly_within=bool(inside and not old_inside),
                                       newly_outside=bool(old_inside and not inside),
                                       changed_vs_common=gained+removed,
                                       added_vs_common=gained,removed_vs_common=removed,
                                       changed_vs_unadjusted=change_raw))
            history[year] = annual_gap
    final = pd.DataFrame(result)
    assert len(final)==len(STATES)*12*len(HISTORICAL_TOLERANCES)
    output = target/"historical_margin_sensitivity.csv"
    temp = output.with_suffix(".csv.tmp")
    final.to_csv(temp,index=False)
    temp.replace(output)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    main(parser.parse_args().root.resolve())
