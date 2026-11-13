# Temporal Fairness Audit

Code for *When Pooled Fairness Audits Fail: Period-Wise Decomposition and Delay-Aware
Monitoring of Group Approval Gaps in Retrained Classifiers* (W. Qiang and M. Hwang, submitted to IEEE Access).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd experiments
```

ACS PUMS files are downloaded from census.gov by Folktables on the first run into
`acs_data/` (not included in this repository).

## Reproducing the results

| Paper item | Command (run inside `experiments/`) | Output |
| --- | --- | --- |
| Smoke test (no download) | `python run_temporal_audit.py --synthetic` | `results_synthetic/` |
| Fig. 2(b), Tables IV-V | `python run_temporal_audit.py --states CA --group race --task income --label-mode quantile` | `results_income_race_CA_d1_quantile/` |
| Table V, nominal / real label | add `--label-mode nominal` or `--label-mode real` | |
| Table V, two-year delay | add `--label-delay 2` | |
| Fig. 2(c) | `python run_temporal_audit.py --states TX NY FL --group race --task income --label-mode quantile` | |
| Fig. 2(d) | `python run_temporal_audit.py --states CA --group race --task employment` | |
| Fig. 2(e) | `python run_temporal_audit.py --states CA --group sex --task income --label-mode quantile` | |
| Stable control | `python run_temporal_audit.py --states CA --group sex --task employment` | |
| Table V, weights and occupation | `python robustness_checks.py` | `robustness_income_race_CA/` |
| Tables VI, IX (synthetic rows) | `python synthetic_validation.py --reps 20` | `synthetic_validation/` |
| Tables VII, IX, Fig. 4 | `python delay_aware_monitor.py --states CA --group race --task income --label-mode quantile --domain-clf lr` | `monitor_*/` |
| Table VII, d = 2 | add `--label-delay 2` | |
| Table VII, employment | `python delay_aware_monitor.py --states CA --group race --task employment --domain-clf lr` | |
| Table A6 (approval share q) | `pooled.csv` of the five audit runs above, renamed and copied to `manuscript/qsens/` | `manuscript/qsens/*.csv` |
| Table III, IV (collected) | `python collect_numbers.py` | printed |
| Tables VII, VIII, A3, A5; Table V (last rows); Fig. 4 | `python run_census_suite.py` (runs `monitor_ablations.py`, `audit_checks.py`, `robustness_checks.py`) | `census_suite/` |
| Tables VI (E3u), A4; Section VI-F label-shift check | `python synthetic_ablations.py --reps 20` and `python synthetic_ablations.py --reps 20 --parts 2 --out synthetic_ablations_p2` | `synthetic_ablations*/` |

All runs use fixed seeds (`--seed 0`). Runtimes on an Apple Silicon laptop: 60-400 s per command.

## Manuscript tables and figures

`manuscript/results_data.py` holds the numbers of the main runs, copied from the
outputs above (see its header); `manuscript/collect_results.py census_suite synthetic_ablations synthetic_ablations_p2`
reads these outputs directly into `collected_tables.json` and `collected_numbers.json`. `python make_tables.py` and `python make_figures.py` build the tables and
Figs. 1-4 (`manuscript/figures/`).

## Corporate credit case (Section VII)

The CSMAR data are licensed and cannot be redistributed. Appendix B and Table A7 of the paper
document the data construction; the CSMAR processing code is available from the author on request.

## Software

Python 3.14.0, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.7.0, Folktables 0.0.12,
pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.9. Logistic-regression and LightGBM results are
identical across machines; XGBoost builds histograms in parallel and its results differ slightly
across hardware (yearly gaps by up to about 0.3 percentage points), without changing any conclusion.

## License

MIT
