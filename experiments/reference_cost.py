"""Paired annual change in discrimination and rejection when H is held out."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_influences(score, outcome):
    pos = score[outcome == 1]
    neg = score[outcome == 0]
    p, n = np.sort(pos), np.sort(neg)
    left = np.searchsorted(n, pos, side="left")
    right = np.searchsorted(n, pos, side="right")
    vpos = (left + right) / (2.0 * len(n))
    left = np.searchsorted(p, neg, side="left")
    right = np.searchsorted(p, neg, side="right")
    vneg = 1 - (left + right) / (2.0 * len(p))
    auc = vpos.mean()
    np.testing.assert_allclose(auc, roc_auc_score(outcome, score), atol=1e-12)
    return auc, vpos, vneg


def main(root):
    source = root / "data" / "scores"
    rows = []
    for mp in sorted((source / "monitor").glob("*/*.npz")):
        rel = mp.relative_to(source / "monitor")
        ap = source / "audit" / rel
        if not ap.exists():
            raise FileNotFoundError(ap)
        with np.load(mp) as m, np.load(ap) as a:
            np.testing.assert_array_equal(m["g"], a["g"])
            y = m["y"]
            np.testing.assert_array_equal(y, a["y"])
            av, apc, anc = auc_influences(a["s"], y)
            mv, mpc, mnc = auc_influences(m["s"], y)
            se = np.sqrt(np.var(mpc-apc, ddof=1)/len(mpc) + np.var(mnc-anc, ddof=1)/len(mnc))
            frr = []
            for run in [a, m]:
                approved = run["s"] >= np.quantile(run["s"], .25)
                frr.append(100 * (1-approved[y == 1].mean()))
            diff = mv - av
            rows.append(dict(config=rel.parent.name,year=int(mp.stem),n=len(y),auc_audit=av,
                             auc_monitor=mv,auc_delta=diff,auc_delta_lo=diff-1.96*se,
                             auc_delta_hi=diff+1.96*se,frr_audit_pp=frr[0],frr_monitor_pp=frr[1],
                             frr_delta_pp=frr[1]-frr[0]))
    if not rows:
        raise ValueError("No matched score files")
    out = root / "results" / "matched"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "reference_holdout_cost.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().root)
