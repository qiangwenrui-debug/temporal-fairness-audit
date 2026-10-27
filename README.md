# Temporal Fairness Audit

Code for *When Pooled Fairness Audits Fail: Period-Wise Decomposition and Delay-Aware
Monitoring of Group Approval Gaps in Retrained Classifiers* (W. Qiang, submitted to IEEE Access).

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
| Table VI | `python synthetic_validation.py --reps 20` | `synthetic_validation/` |
| Table VII, Fig. 4 | `python delay_aware_monitor.py --states CA --group race --task income --label-mode quantile --domain-clf lr` | `monitor_*/` |
| Table VII, d = 2 | add `--label-delay 2` | |
| Table VII, employment | `python delay_aware_monitor.py --states CA --group race --task employment --domain-clf lr` | |
| Table III, IV (collected) | `python collect_numbers.py` | printed |

All runs use fixed seeds (`--seed 0`). Runtimes on an Apple Silicon laptop: 60-400 s per command.

## Manuscript tables and figures

`manuscript/results_data.py` holds every number reported in the paper, transcribed from the
outputs above. `python make_tables.py` and `python make_figures.py` build the tables and
Figs. 1-4; `node build.js` (npm package `docx`) assembles the Word draft.

## Corporate credit case (Section VII)

The CSMAR data are licensed and cannot be redistributed. The numbers for this case are
taken from the author's earlier analysis of the same data, and the CSMAR processing code
is available from the author on request.

## Software

Python 3.14.0, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.7.0, Folktables 0.0.12,
pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.9. XGBoost results can differ in the second or
third decimal place across processor architectures.

## License

MIT
