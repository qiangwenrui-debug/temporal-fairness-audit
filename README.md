# Temporal fairness audit code

Code for W. Qiang and M. Huang, "When Pooled Fairness Audits Fail: Approval Gaps in Annually Retrained Classifiers" (submitted to IEEE Access). The full replication package, with data, fitted models, scores and results, is archived at https://doi.org/10.5281/zenodo.24394770.

The replication guide (`README.md` at the top of the replication package) lists the commands for the figures, tables, four-state tests and monitoring results in the paper. Run those commands from the directory containing `code/`, `data/` and `results/`.

## Historical census audits and simulations

The commands below run from `code/experiments/` and generate the historical census and simulation results. The census tasks use American Community Survey one-year person files, downloaded through Folktables. The simulation uses generated data.

| Analysis | Command |
|---|---|
| California income by race | `python run_temporal_audit.py --states CA --group race --task income --label-mode quantile` |
| Texas, New York and Florida, pooled within each year | `python run_temporal_audit.py --states TX NY FL --group race --task income --label-mode quantile` |
| California employment by race | `python run_temporal_audit.py --states CA --group race --task employment` |
| California income by sex | `python run_temporal_audit.py --states CA --group sex --task income --label-mode quantile` |
| California employment by sex | `python run_temporal_audit.py --states CA --group sex --task employment` |
| Survey weights and occupation sensitivity | `python robustness_checks.py` |
| Simulated distribution shifts | `python synthetic_validation.py --reps 20` |
| Score misspecification and reference sample size | `python synthetic_ablations.py --reps 20` |
| Label shift with group-specific features | `python synthetic_ablations.py --reps 20 --parts 2 --out synthetic_ablations_p2` |

For the alternative income cutoffs use `--label-mode nominal` or `--label-mode real`. For a two-year outcome delay use `--label-delay 2`. The census audit and the residual monitor use different training cutoffs. The monitor holds back the most recent labelled year as a reference cohort.

## Software and data

The historical census and simulation runs used Python 3.14.0 on Apple Silicon. Package versions for these analyses are in `requirements.txt`. The four-state runs used Python 3.12.10 on Windows. Their versions and data preparation commands are in the monitoring guide (`results/monitor/README.md` in the replication package).

Appendix B documents the corporate credit cohort. The replication package provides the public disclosure dates and aggregate results for that example. The underlying corporate statements are licensed by CSMAR.

## License

MIT
