"""
Numbers of the main runs, read by make_tables.py and make_figures.py. They were copied from the
output files of the commands listed in README.md (runs of 27 Oct 2026, seed 0, XGBoost unless a
block says otherwise); proportions are stored as fractions (0.05 = 5 pp), q = 0.75 throughout.
Rerunning a command reproduces the corresponding block (XGBoost to within about 0.3 pp across
hardware, see README).

Block          Paper item                   Source
YEARLY         Figs. 2(a-e), 3, Table A1    census: results_<setting>/yearly.csv (run_temporal_audit.py)
LABEL_ROBUST   Fig. 2(f)                    results_income_race_CA_d{1,2}_{nominal,real,quantile}/yearly.csv
POOLED         Table IV                     results_<setting>/pooled.csv and carryover.csv
ROBUST         Table V (first rows)         the four LABEL_ROBUST runs, pooled.csv
SYNTH          Table VI                     synthetic_validation/ (synthetic_validation.py --reps 20)
SENS           Table IX                     synthetic_validation/ and monitor_*/monitor_summary.md
"csmar" keys   Table IV, Figs. 2(a), 3      outputs of the CSMAR pipeline (Table A8: timing_check.csv);
                                            the data are licensed, so this pipeline is not in the
                                            repository (available on request)

The remaining table cells (Tables V-VIII, A3-A5 and Fig. 4) are not stored here: collect_results.py
reads them directly from the outputs of run_census_suite.py and synthetic_ablations.py into
collected_tables.json and collected_numbers.json.
"""

ACS_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
CSMAR_YEARS = list(range(2014, 2023))

# ---------------------------------------------------------------- yearly, XGBoost (Figs. 2-3, Table A1)
YEARLY = {
    "csmar": dict(
        label="Corporate credit (CSMAR)", years=CSMAR_YEARS,
        G=[-.159, -.165, -.127, -.038, .051, .049, .043, .081, .070],
        G_lo=[-.201, -.207, -.164, -.073, .020, .016, .012, .052, .043],
        G_hi=[-.124, -.126, -.089, -.003, .085, .081, .071, .108, .098],
        RG=[-.147, -.157, -.123, -.041, .042, .026, .014, .045, .028],
        beta=[.662, .757, .619, 0, .012, 0, 0, .403, .365],
        pooled_beta=0.0, pooled_G=-.0083),
    "inc_race_ca": dict(
        label="Income × race, CA", years=ACS_YEARS,
        G=[-.0273, -.0402, -.0383, -.0277, -.0486, -.1119, -.1103, -.1216],
        G_lo=[-.0375, -.0478, -.0467, -.0374, -.0586, -.1245, -.1215, -.1313],
        G_hi=[-.0192, -.0299, -.0274, -.0182, -.0382, -.1021, -.0996, -.1112],
        RG=[-.0047, -.0077, -.0041, -.0067, -.0140, -.0325, -.0327, -.0281],
        RC=[-.0227, -.0325, -.0342, -.0211, -.0346, -.0794, -.0777, -.0935],
        beta=[0, 0, 0, 0, 0, .586, .590, .629],
        pi1=[.3463, .3191, .3221, .3346, .3226, .3399, .3359, .3425],
        pi0=[.4290, .4183, .4343, .4325, .4317, .4871, .4970, .4955],
        pooled_beta=.245, pooled_G=-.0643),
    "inc_race_tx": dict(
        label="Income × race, TX/NY/FL", years=ACS_YEARS,
        G=[-.1140, -.1048, -.1088, -.1088, -.1100, -.1456, -.1377, -.1359],
        G_lo=[-.1192, -.1100, -.1124, -.1136, -.1161, -.1502, -.1426, -.1407],
        G_hi=[-.1088, -.1002, -.1040, -.1031, -.1048, -.1402, -.1326, -.1302],
        RG=[-.0240, -.0213, -.0248, -.0220, -.0265, -.0398, -.0337, -.0290],
        beta=[.602, .553, .575, .569, .581, .701, .681, .662],
        pi1=[.2748, .2882, .2856, .2904, .2827, .2954, .2955, .2853],
        pi0=[.4133, .4243, .4247, .4236, .4218, .4601, .4520, .4358],
        pooled_beta=.622, pooled_G=-.1200),
    "emp_race_ca": dict(
        label="Employment × race, CA", years=ACS_YEARS,
        G=[-.0420, -.0360, -.0449, -.0532, -.0582, -.0802, -.0740, -.0631],
        G_lo=[-.0488, -.0423, -.0526, -.0607, -.0646, -.0883, -.0798, -.0701],
        G_hi=[-.0357, -.0290, -.0384, -.0465, -.0524, -.0737, -.0644, -.0573],
        RG=[-.0003, -.0005, -.0008, .0000, -.0010, -.0025, -.0007, -.0011],
        beta=[0, 0, 0, .071, .134, .392, .337, .227],
        pi1=[.3755, .3821, .3873, .3930, .3959, .3651, .3890, .3964],
        pi0=[.4433, .4494, .4547, .4585, .4636, .4363, .4514, .4508],
        pooled_beta=.113, pooled_G=-.0559),
    "inc_sex_ca": dict(
        label="Income × sex, CA", years=ACS_YEARS,
        G=[-.0655, -.0643, -.0654, -.0650, -.0650, -.0549, -.0511, -.0482],
        G_lo=[-.0694, -.0687, -.0694, -.0682, -.0688, -.0590, -.0549, -.0519],
        G_hi=[-.0614, -.0607, -.0611, -.0609, -.0609, -.0513, -.0472, -.0446],
        RG=[.0001, -.0005, .0001, .0003, -.0022, .0011, -.0009, .0025],
        beta=[.229, .223, .232, .232, .221, .086, .019, 0],
        pi1=[.3360, .3236, .3379, .3385, .3399, .3366, .3472, .3480],
        pi0=[.4470, .4322, .4479, .4548, .4528, .4341, .4449, .4449],
        pooled_beta=.162, pooled_G=-.0598),
}

# label-definition robustness, CA income x race, XGBoost yearly G (Fig. 2(f))
LABEL_ROBUST = {
    "Nominal $50k": dict(years=ACS_YEARS,
                         G=[-.0251, -.0358, -.0367, -.0276, -.0504, -.1091, -.1078, -.1205]),
    "Real $50k (2015 $)": dict(years=ACS_YEARS,
                               G=[-.0268, -.0362, -.0374, -.0249, -.0470, -.1087, -.1083, -.1212]),
    "Top 40% within year": dict(years=ACS_YEARS,
                                G=[-.0273, -.0402, -.0383, -.0277, -.0486, -.1119, -.1103, -.1216]),
    "Top 40%, d = 2": dict(years=ACS_YEARS[1:],
                           G=[-.0383, -.0378, -.0257, -.0459, -.1119, -.1085, -.1203]),
}

# ---------------------------------------------------------------- pooled summary (Table IV)
# (pooled gap, CI lo, CI hi, mean|G|, years outside +-5, n years, beta min, beta max, pooled beta,
#  carry fails, carry n, carry max excess gap)
POOLED = {
    "csmar": {
        "LR":   (-.0586, -.081, -.035, .0936, 5, 9, 0, .736, .110, 2, 8, .057),
        "XGB":  (-.0083, -.027, .012, .0869, 6, 9, 0, .757, 0, 3, 8, .081),
        "LGBM": (-.0060, -.025, .015, .0879, 7, 9, 0, .697, 0, 3, 8, .077)},
    "inc_race_ca": {
        "LR":   (-.0735, -.0768, -.0703, .0752, 4, 8, 0, .688, .349, 4, 7, .1017),
        "XGB":  (-.0643, -.0679, -.0602, .0657, 3, 8, 0, .629, .245, 3, 7, .1119),
        "LGBM": (-.0626, -.0659, -.0594, .0640, 3, 8, 0, .633, .219, 2, 7, .1094)},
    "inc_race_tx": {
        "LR":   (-.1372, -.1391, -.1353, .1380, 8, 8, .627, .740, .681, 3, 7, .0656),
        "XGB":  (-.1200, -.1217, -.1179, .1207, 8, 8, .553, .701, .622, 3, 7, .0674),
        "LGBM": (-.1178, -.1194, -.1159, .1185, 8, 8, .542, .695, .612, 3, 7, .0691)},
    "emp_race_ca": {
        "LR":   (-.0743, -.0768, -.0723, .0751, 8, 8, .006, .566, .355, 4, 7, .0706),
        "XGB":  (-.0559, -.0581, -.0535, .0564, 5, 8, 0, .392, .113, 3, 7, .0707),
        "LGBM": (-.0549, -.0576, -.0525, .0554, 5, 8, 0, .362, .092, 3, 7, .0690)},
    "inc_sex_ca": {
        "LR":   (-.0408, -.0420, -.0394, .0409, 0, 8, 0, 0, 0, 0, 7, .0471),
        "XGB":  (-.0598, -.0612, -.0585, .0599, 7, 8, 0, .232, .162, 1, 7, .0506),
        "LGBM": (-.0595, -.0608, -.0583, .0597, 6, 8, 0, .258, .165, 3, 7, .0515)},
}
PATTERN = {"csmar": "Cancellation", "inc_race_ca": "Level break",
           "inc_race_tx": "High level + step", "emp_race_ca": "Drift + peak",
           "inc_sex_ca": "Convergence"}

# ---------------------------------------------------------------- robustness (Table V, first rows)
# label, delay, mean gap 2015-19 (2016-19 for d=2), mean gap 2021-23, pooled gap, beta max, pooled beta
ROBUST = [
    ("Nominal $50k", 1, -.0351, -.1125, -.0627, .607, .223),
    ("Real $50k (2015 $)", 1, -.0345, -.1127, -.0624, .628, .234),
    ("Top 40% within year", 1, -.0364, -.1146, -.0643, .629, .245),
    ("Top 40% within year", 2, -.0369, -.1136, -.0683, .616, .305),
]

# ---------------------------------------------------------------- synthetic (Table VI)
# scenario: G first,last ; RG first,last ; RC first,last ; bias E0,E1,E2,E3,E3_oracle ; MAE E0..E3_oracle
SYNTH = {
    "No drift": ((-12.25, -12.03), (-3.98, -3.79), (-8.27, -8.24),
                 (.01, .06, .18, -.02, 0), (.97, .65, .91, .98, .97)),
    "Covariate shift": ((-20.38, -45.54), (-6.85, -17.27), (-13.53, -28.27),
                        (1.74, .02, .45, .11, .04), (2.11, 1.10, 1.49, 1.54, 1.54)),
    "Label shift": ((-2.05, -7.76), (-.03, .01), (-2.02, -7.76),
                    (0, -.33, .02, -.05, -.05), (.33, .38, .30, .34, .34)),
    "Concept shift": ((-12.09, -11.87), (-2.56, -.71), (-9.52, -11.16),
                      (-.32, -1.99, -.08, -.30, -.30), (1.02, 2.00, .92, 1.01, 1.02)),
}

# ---------------------------------------------------------------- concept-shift sensitivity (Table IX)
# setting: (history-calibrated: covered, n, mean width pp), (gamma 1.1 cov, width), (1.25 cov, width), max gamma_needed
SENS = {
    "Synthetic, no drift":        ((0.717, None, 3.01), (0.957, 4.72), (1.000, 11.22), None),
    "Synthetic, covariate shift": ((0.692, None, 4.44), (0.864, 5.94), (0.993, 14.03), None),
    "Synthetic, label shift":     ((0.725, None, 1.10), (0.400, 0.49), (0.793, 1.16), None),
    "Synthetic, concept shift":   ((0.692, None, 3.14), (0.893, 4.15), (1.000, 9.90), None),
    "CA income × race, d = 1":    ((5, 6, 1.22), (1.0, 1.89), (1.0, 4.53), 1.061),
    "CA income × race, d = 2":    ((3, 4, 2.80), (0.833, 1.95), (1.0, 4.66), 1.138),
    "CA employment × race, d = 1": ((5, 6, 0.48), (1.0, 0.52), (1.0, 1.25), 1.098),
}
