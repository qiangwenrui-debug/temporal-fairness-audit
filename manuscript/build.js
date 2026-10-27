// Builds manuscript_IEEE_Access_draft.docx
const fs = require("fs");
const L = require("./lib");
const { d, FONT, SANS, BLUE, PAGE_W, MARGIN_LR, COL_GAP, FULL_W, COL_W, runs, body, h1, h2, item,
  mr, sub, sup, subsup, frac, sum, equation, figure, tableCaption, tableNote, table } = L;
const { Paragraph, TextRun, AlignmentType, SectionType, Header, Footer, PageNumber, Table, TableRow,
  TableCell, WidthType, BorderStyle, Tab, TabStopType } = d;
const T = JSON.parse(fs.readFileSync("tables.json"));

const sections = [];
let cur = null;
function section(cols, first = false) {
  cur = { cols, children: [], first };
  sections.push(cur);
}
const add = (...xs) => xs.flat().forEach((x) => cur.children.push(x));

// =========================================================== FRONT MATTER (one column)
section(1, true);
const small = (t, o = {}) => new Paragraph({ spacing: { after: 20 }, ...o.p,
  children: runs(t, { size: 14, font: SANS, color: "444444", ...o.r }) });
add(small("Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000."),
  small("Digital Object Identifier 10.1109/ACCESS.2026.DOI", { p: { spacing: { after: 240 } } }));
add(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ font: SANS, size: 40, bold: true, color: "1A1A1A",
  text: "When Pooled Fairness Audits Fail: Period-Wise Decomposition and Delay-Aware Monitoring of Group Approval Gaps in Retrained Classifiers" })] }));
add(new Paragraph({ spacing: { after: 80 }, children: [
  new TextRun({ text: "WENRUI QIANG", font: SANS, size: 20, bold: true }),
  new TextRun({ text: "1", font: SANS, size: 20, superScript: true }),
  new TextRun({ text: "  [TBD: confirm name order and IEEE membership grade]", font: SANS, size: 16, highlight: "yellow" })] }));
add(small("^{1}[TBD: Department, University, City, Postcode, Country] (e-mail: qiangwenrui@gmail.com) [VERIFY corresponding e-mail]"),
  small("Corresponding author: Wenrui Qiang (e-mail: qiangwenrui@gmail.com)."),
  small("[TBD: funding statement, e.g., This work received no external funding.]", { p: { spacing: { after: 200 } } }));

const ABSTRACT = "Post-processing threshold calibration keeps the gap in approval rates between groups within a tolerance, but its strength is usually set once on pooled data while the scoring model is retrained and group risk profiles drift. We study when such pooled audits fail and how the gap can be monitored when outcome labels arrive late. We formalize two exact decompositions: a pooled gap equals a weighted average of period gaps plus a composition term, and each period gap equals a residual gap among qualified individuals plus a risk-composition component. We propose a delay-aware estimator of the residual gap that applies the current decision rule to the latest labelled period, reweighted for covariate shift. With yearly retraining on U.S. census microdata (2014–2023; two tasks, two group attributes, four states) and corporate credit data, we identify four failure modes of pooled audits: cancellation, level break, gradual drift and convergence. In California income prediction, the Black–White approval gap widened from about −3 to −12 percentage points after 2019, and the pooled calibration strength (0.25) matched no year, whose requirements ranged from 0 to 0.63. The break persists under three income-label definitions, one- and two-year label delays, three model classes and survey weighting. At the 2021 break in California, the proposed estimator recovered 88% of the realized residual before labels arrived, against 46% for the last observed value. Fairness audits of retrained classifiers should be performed per period and report both components.";
const ruleP = (top) => ({ border: top ? { top: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } }
  : { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } } });
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 250 }, ...ruleP(true),
  children: [new TextRun({ text: "ABSTRACT ", font: SANS, size: 18, bold: true, color: BLUE }), ...runs(ABSTRACT, { size: 18 })] }));
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 }, ...ruleP(false),
  children: [new TextRun({ text: "INDEX TERMS ", font: SANS, size: 18, bold: true, color: BLUE }),
    ...runs("Algorithmic fairness, concept drift, covariate shift, delayed labels, distribution shift, model monitoring, post-processing, threshold calibration.", { size: 18 })] }));

// =========================================================== I. INTRODUCTION
section(2);
add(h1("I. Introduction"));
add(body("Machine-learning scores now drive many screening decisions, from credit approval to the allocation of services, and auditors increasingly ask whether the resulting decisions treat groups evenly [3], [5], [6]. For a model already in deployment, the least intrusive remedy is post-processing: the score is left unchanged and group-specific thresholds bring the difference in approval rates between groups within a tolerance [1], [2]. Regulatory guidance stresses that such properties must be monitored after deployment, not certified once [4], [33].", { noIndent: true }));
add(body("In practice, the calibration strength is often chosen on a pooled sample of several periods and then reused. This treats the approval gap as a stable property of the model. Deployed scoring models, however, are retrained on recent data, and the relative risk of groups changes with economic conditions. A model trained on earlier periods ranks groups according to the earlier relationship, so its gap moves with a lag. Surveys of fairness under distribution shift note that guarantees established at one time need not survive changes in the data-generating process [13]–[17], but evidence on how group gaps and their remedies behave across successive deployment periods remains scarce."));
add(body("A second difficulty is timing. The approval gap depends only on scores and group membership, so it can be computed at decision time. Error-rate quantities, such as the gap in approval rates among qualified individuals, require outcome labels, which in credit, hiring or income prediction arrive months or years later. An auditor therefore sees the part of the gap that is cheap to fix but not the part that indicates unequal treatment of equally qualified people, until it is too late to act on that period."));
add(body("This paper addresses both problems. We ask (i) when a pooled audit and a single calibration parameter fail to represent individual periods, (ii) which component of a period gap is responsible, and (iii) whether the residual component can be estimated before labels arrive. Our contributions are as follows."));
add(item("1)", "**A period-wise decomposition framework.** We formalize two exact identities: a pooled approval gap equals a weighted average of period gaps plus a composition term (Proposition 1), and each period gap equals a residual gap among qualified individuals plus a risk-composition component (Proposition 2). We show that the bound on the risk component is informative only when unfavourable outcomes are rare, and we give an audit procedure with O(*n* log *n*) cost per period (Algorithm 1)."),
  item("2)", "**A delay-aware residual estimator.** We estimate the residual gap at decision time by applying the current decision rule to the most recent labelled period, importance-weighted towards the current covariate distribution with a cross-fitted domain classifier. We compare it with the last observed value and with two soft-label estimators and characterize when each is unbiased."),
  item("3)", "**A taxonomy of audit failures across settings.** Using yearly retraining on U.S. census microdata for 2014–2023 [18] (income and employment tasks, race and sex groups, four states) and a corporate credit data set, we identify four failure modes of pooled audits (cancellation, level break, gradual drift and convergence) and a stable control. The results hold across three income-label definitions, two label delays and three model classes."),
  item("4)", "**Reproducible code** that regenerates every table and figure from public data [TBD: repository link]."));
add(body("The main empirical findings are as follows. In California income prediction, the approval gap between Black and White individuals under gradient boosting stayed within −2.7 to −4.9 percentage points (pp) in 2015–2019 and widened to −11.0 to −12.2 pp in 2021–2023. The calibration strength computed on the pooled sample (0.25) matched no year: the per-year requirement was 0 before the break and 0.59–0.63 after it. In corporate credit data, by contrast, yearly gaps of −16.5 to +8.1 pp cancelled to a pooled gap of −0.8 pp. Shortening the label delay from two years to one did not remove the break. At that break, the proposed estimator recovered 88% of the realized residual before labels arrived, whereas the last observed value recovered 46%."));
add(body("Section II reviews related work. Section III presents the framework and Section IV the delay-aware estimator. Section V describes the experimental setup and Section VI the results. Section VII presents the corporate credit case, Section VIII discusses implications and limitations, and Section IX concludes."));

// =========================================================== II. RELATED WORK
add(h1("II. Related Work"));
add(h2("A. Group Fairness Criteria and Post-Processing"));
add(body("Common criteria include statistical parity, equality of opportunity and calibration within groups. Hardt *et al.* [1] showed that equality of opportunity can be attained by group-specific thresholds on an existing score. Kleinberg *et al.* [7] and Chouldechova [8] proved that calibration within groups and balanced error rates cannot in general hold together when base rates differ, and Pleiss *et al.* [9] characterized the relaxations compatible with calibration. Corbett-Davies and Goel [10] noted that outcome gaps can arise entirely from differences in risk distributions. An approval gap alone therefore cannot separate risk-based differences from unequal treatment, which motivates our decomposition. Kozodoi *et al.* [2] compared pre-, in- and post-processing on credit data, all on a single split.", { noIndent: true }));
add(h2("B. Fairness Under Distribution Shift"));
add(body("A growing literature asks whether fairness established on training data transfers to deployment data. Singh *et al.* [14] and Rezaei *et al.* [15] studied fairness under covariate shift, Maity *et al.* [16] analysed subpopulation shift, and Schrouff *et al.* [17] diagnosed failures of fairness transfer in clinical settings; Barrainkua *et al.* [13] survey the area. These works typically compare one source and one target distribution and focus on training-time interventions. We instead study a sequence of retraining periods, the behaviour of a post-processing parameter along that sequence, and how pooled audits aggregate it.", { noIndent: true }));
add(h2("C. Long-Term and Dynamic Fairness"));
add(body("Liu *et al.* [11] showed that fairness constraints must be judged by their effects over time, and D’Amour *et al.* [12] used simulations, including a lending example, to show that statically evaluated interventions can have different long-run consequences. These studies model feedback between decisions and populations. Our setting has no feedback by construction; the drift is exogenous, which isolates the measurement question of what a pooled audit can and cannot certify. Lum *et al.* [25] showed that fairness metrics computed on finite samples are noisy and biased, which motivates our interval estimates.", { noIndent: true }));
add(h2("D. Label-Free Performance Estimation and Shift Detection"));
add(body("When labels are delayed, model performance can be estimated from unlabelled target data under assumptions about the shift. Importance weighting corrects for covariate shift [19], [20], with density ratios estimated by kernel mean matching [35] or by a discriminative domain classifier [36]. Label-shift corrections assume instead that *P*(*x* | *y*) is stable [21]. Rabanser *et al.* [22] compared methods for detecting shift, Garg *et al.* [23] predicted out-of-distribution accuracy from confidence scores, and Gama *et al.* [24] survey concept-drift adaptation. To our knowledge, these tools have not been used to estimate a conditional fairness gap before labels arrive, which is the role of our delay-aware estimator.", { noIndent: true }));
add(h2("E. Fairness in Credit Scoring"));
add(body("Fuster *et al.* [5] found that more flexible models shift gains and losses between groups in U.S. mortgages, Das *et al.* [6] review algorithmic fairness in finance, and Hurlin *et al.* [26] develop tests of whether credit-scoring gaps are explained by creditworthiness. John [29] reported that fairness metrics of a credit model deteriorate under concept drift. Our corporate credit case extends this line to an industry grouping whose relative risk reverses over the business cycle [34]. Table I positions this study against the closest work.", { noIndent: true }));
add(tableCaption("I", "Positioning Relative to Closely Related Work"));
add(table(["Study", "Several deployment periods", "Separates risk and residual", "Calibration parameter over time", "Label delay"],
  [["Hardt et al. [1]", "No", "No", "No", "No"], ["Kozodoi et al. [2]", "No", "No", "No", "No"],
   ["Liu et al. [11]; D’Amour et al. [12]", "Yes (simulated)", "No", "No", "No"],
   ["Shift-robust fairness [13]–[17]", "Source–target", "No", "No", "No"],
   ["Garg et al. [23]", "Source–target", "No", "No", "Yes (accuracy)"],
   ["Hurlin et al. [26]", "No", "Tests", "No", "No"],
   ["**This study**", "**Yes (8–9 years)**", "**Yes (exact)**", "**Yes**", "**Yes (fairness)**"]],
  [1640, 900, 850, 850, 800], { size: 14 }));
add(tableNote("“Source–target”: one training and one deployment distribution."));

// =========================================================== III. FRAMEWORK
add(h1("III. Period-Wise Decomposition Framework"));
add(h2("A. Setting and Notation"));
add(body("Periods are indexed by *t* = 1, …, *T*. In period *t*, individual *i* has features *x*_{i}, a binary group attribute *g*_{i} ∈ {0, 1} and a binary outcome *y*_{i}, where *y* = 1 is the favourable outcome (“qualified”). A model *m*_{t}, trained on labelled data available before *t*, outputs a score *s*_{i} = *m*_{t}(*x*_{i}); the group attribute is not a model input. A screen approves the share *q* of individuals with the highest scores, so the global threshold *T*_{t}(*q*) is the (1 − *q*) quantile of the period-*t* scores and *T*_{gt}(*q*) is the corresponding within-group quantile. Threshold calibration with strength β ∈ [0, 1] uses", { noIndent: true }));
add(equation([sub("τ", "gt"), mr("(β)=(1−β)"), sub("T", "t"), mr("(q)+β"), sub("T", "gt"), mr("(q)")], 1));
add(body("and approves individual *i* if *s*_{i} ≥ τ_{gt}(β) for its group. Let *P*_{gt}(β) be the approval rate of group *g* and *G*_{t}(β) = *P*_{1t}(β) − *P*_{0t}(β) the approval gap. Given a tolerance δ, the minimum calibration strength is", { noIndent: true }));
add(equation([subsup("β", "t", "*"), mr("=min{β∈[0,1]:|"), sub("G", "t"), mr("(β)|≤δ}.")], 2));
add(body("*G*_{t}(β) need not be monotone in β, so β^{*}_{t} is found by search over a grid (Algorithm 1). The pooled strength β^{*}_{pool} is the smallest β whose pooled gap, with the same β applied in every period, lies within δ.", { noIndent: true }));
add(h2("B. Proposition 1: Temporal Aggregation"));
add(body("Let *n*_{gt} be the number of group-*g* individuals in period *t* and *w*_{gt} = *n*_{gt} / Σ_{s} *n*_{gs}. The pooled approval rate of group *g* is *P*_{g} = Σ_{t} *w*_{gt}*P*_{gt}.", { noIndent: true }));
add(body("**Proposition 1.** The pooled approval gap satisfies", { noIndent: true }));
add(equation([sub("G", "pool"), mr("="), sum("t", [sub("w", "1t"), sub("G", "t")]), mr("+"), sum("t", [mr("("), sub("w", "1t"), mr("−"), sub("w", "0t"), mr(")"), sub("P", "0t")]), mr(".")], 3));
add(body("*Proof.* Add and subtract Σ_{t} *w*_{1t}*P*_{0t} in *P*_{1} − *P*_{0}. ∎", { noIndent: true }));
add(body("The first term is a weighted average of period gaps; the second, a composition term, vanishes when both groups are distributed identically over periods. Because |Σ_{t} *w*_{1t}*G*_{t}| ≤ Σ_{t} *w*_{1t}|*G*_{t}|, with equality only when all period gaps share a sign, a pooled audit can understate the typical period gap by an arbitrary factor when gaps change sign (cancellation). Even without sign changes, a pooled gap lies between the period gaps and need not represent any of them (level break). The relevant audit statistics are therefore the dispersion of period gaps and the number and size of exceedances of the tolerance."));
add(h2("C. Proposition 2: Risk Composition and Residual"));
add(body("Within a period, let π_{g} = *P*(*y* = 1 | *g*) and let *a*_{gy} be the approval rate of group-*g* individuals with outcome *y*.", { noIndent: true }));
add(body("**Proposition 2.** The period gap decomposes exactly as", { noIndent: true }));
add(equation([sub("G", "t"), mr("="), sub("RG", "t"), mr("+"), sub("RC", "t"), mr(",  "), sub("RG", "t"), mr("="), sub("a", "11"), mr("−"), sub("a", "01")], 4));
add(equation([sub("RC", "t"), mr("=(1−"), sub("π", "1"), mr(")("), sub("a", "10"), mr("−"), sub("a", "11"), mr(")−(1−"), sub("π", "0"), mr(")("), sub("a", "00"), mr("−"), sub("a", "01"), mr(")")], 5));
add(body("If the screen weakly separates outcomes within each group (*a*_{g0} ≤ *a*_{g1} for *g* = 0, 1), then |*RC*_{t}| ≤ max(1 − π_{1}, 1 − π_{0}).", { noIndent: true }));
add(body("*Proof.* Substitute *P*_{g} = π_{g}*a*_{g1} + (1 − π_{g})*a*_{g0} into *P*_{1} − *P*_{0} and rearrange. Under the separation condition, each term (1 − π_{g})(*a*_{g0} − *a*_{g1}) lies in [−(1 − π_{g}), 0], so their difference lies in [−(1 − π_{1}), 1 − π_{0}]. ∎", { noIndent: true }));
add(body("*RG*_{t} is the approval gap among qualified individuals, the negative of the gap in false-rejection rates. *RC*_{t} is non-zero only because the groups differ in outcome distribution and the screen separates outcomes. The bound is informative only when unfavourable outcomes are rare: with distress rates of 2.7%–6.1% in 2014–2016 in the corporate credit case (Section VII), at most a few points of any gap can be risk-based, whereas with unfavourable shares of 41%–73% in the census tasks the bound is not binding and the decomposition must be computed. *RG*_{t} is defined relative to an observed label and is not a measure of discrimination: it changes even under pure covariate shift with a stable *P*(*y* | *x*) (Section VI-F)."));
add(h2("D. Audit Procedure"));
add(body("Algorithm 1 summarizes the audit, and Fig. 1 shows how it combines with the estimator of Section IV. The cost per period is O(*n* log *n* + |*B*| log *n*) for *n* scored individuals and a grid *B*, negligible next to model training; we use |*B*| = 1001.", { noIndent: true }));
const ALG = ["Algorithm 1  Period-wise fairness audit",
  "Input: periods 1..T, delay d, share q, tolerance δ, grid B",
  "for t = 1..T:",
  "  train m_t on periods ≤ t − d",
  "  s ← m_t(X_t); sort scores by group    # O(n log n)",
  "  for β in B: G_t(β) by bisection      # O(|B| log n)",
  "  β*_t ← smallest β in B with |G_t(β)| ≤ δ",
  "  estimate RG_t at decision time (Sec. IV)",
  "  when labels arrive: RG_t, RC_t (Prop. 2), CIs",
  "report: G_pool and its decomposition (Prop. 1);",
  "  dispersion of G_t; exceedances of δ;",
  "  β*_t vs. β*_pool; estimated vs. realized RG_t"];
add(new Table({ width: { size: COL_W, type: WidthType.DXA }, columnWidths: [COL_W],
  borders: { top: { style: BorderStyle.SINGLE, size: 8, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000" },
    left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    insideHorizontal: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
  rows: [new TableRow({ children: [new TableCell({ width: { size: COL_W, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: ALG.map((l, i) => new Paragraph({ spacing: { after: 0 }, border: i === 0 ? { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 2 } } : undefined,
      children: [new TextRun({ text: l, font: "Courier New", size: 13, bold: i === 0 })] })) })] })] }));
add(new Paragraph({ spacing: { after: 80 }, children: [] }));

section(1);
add(figure("fig1_pipeline.png", 7.0, 1, "Period-wise audit with delay-aware monitoring. Steps 1–4 run when period-*t* decisions are made and need no outcome labels; steps 5–6 run once labels of period *t* arrive after the delay *d*."));
section(2);

// =========================================================== IV. DELAY-AWARE MONITORING
add(h1("IV. Delay-Aware Residual Monitoring"));
add(body("The approval gap *G*_{T} and the strength β^{*}_{T} can be computed when period-*T* decisions are made. The residual *RG*_{T} cannot, because the outcomes of period *T* are observed only after a delay *d*. We therefore estimate *RG*_{T} at decision time.", { noIndent: true }));
add(h2("A. Information at Decision Time"));
add(body("Let *H* = max{*t* ≤ *T* − *d*} be the latest labelled period. The model *m*_{T} is trained on periods before *H*, and period *H* is held out as a labelled reference set that the model has not seen. At decision time the auditor observes (*x*, *g*, *y*) on *H* and (*x*, *g*) on the current period *D*_{T}, the scores of both, and the current rule τ_{g} = τ_{gT}(β). The target is", { noIndent: true }));
add(equation([sub("RG", "T"), mr("="), sub("Pr", "T"), mr("(s≥"), sub("τ", "g"), mr("|g=1,y=1)−"), sub("Pr", "T"), mr("(s≥"), sub("τ", "g"), mr("|g=0,y=1)")], 6));
add(h2("B. Estimators"));
add(body("Each estimator replaces the unobserved group-wise approval rate among the qualified, *a*_{g1}, by a computable quantity.", { noIndent: true }));
add(item("•", "**E0, last observed value.** The residual of period *H* under the rule of period *H*. It is exact without drift and lags by *d* periods otherwise."),
  item("•", "**E1, soft labels.** The current score estimates *P*(*y* = 1 | *x*): â_{g1} = Σ 1{*s*_{i} ≥ τ_{g}}*s*_{i} / Σ *s*_{i} over group-*g* individuals in *D*_{T}. Unbiased if the model is calibrated within groups in period *T*, which boosted-tree probabilities often are not [28]."),
  item("•", "**E2, recalibrated soft labels.** As E1, after replacing *s* by a group-wise isotonic recalibration fitted on *H* [27]. Unbiased if *P*(*y* | *s*, *g*) is the same in *H* and *T*."),
  item("•", "**E3, covariate-weighted reference (proposed).** Apply the current rule to the scores of *H* and reweight the qualified individuals of *H* towards the current covariate distribution:"));
add(equation([sub("â", "g1"), mr("="), frac([sum("i∈H:g,y=1", [sub("w", "i"), mr("1{"), sub("s", "i"), mr("≥"), sub("τ", "g"), mr("}")])], [sum("i∈H:g,y=1", [sub("w", "i")])]),
  mr(",  "), sub("w", "i"), mr("="), frac([sub("p", "T"), mr("("), sub("x", "i"), mr(","), sub("g", "i"), mr(")")], [sub("p", "H"), mr("("), sub("x", "i"), mr(","), sub("g", "i"), mr(")")])], 7));
add(body("Under covariate shift, *P*_{T}(*y* | *x*, *g*) = *P*_{H}(*y* | *x*, *g*), and with overlapping support the self-normalized ratio inside each (*g*, *y* = 1) cell is consistent for *a*_{g1} in period *T* [19], [20]. E3 uses the labels of *H*, which E1 does not, and the current rule and covariates, which E0 does not."));
add(h2("C. Density-Ratio Estimation"));
add(body("We estimate *w* with a discriminative domain classifier [36]. A classifier *c*(*x*, *g*) is trained to distinguish rows of *D*_{T} from rows of *H*, and *w* = *c*/(1 − *c*) · *n*_{H}/*n*_{T}. To avoid in-sample overconfidence we use two-fold cross-fitting: each row of *H* receives a probability from a classifier fitted on the other fold. Weights are clipped at their 99th percentile and normalized to mean one, and we report the effective sample size (Σ*w*)^{2}/Σ*w*^{2} as a share of *n*_{H}. The classifier is a logistic regression on (*x*, *g*, *x*·*g*), which lets the log ratio differ by group, or gradient-boosted trees.", { noIndent: true }));
add(h2("D. When Each Estimator Works"));
add(body("Table II summarizes the conditions. No label-free estimator is consistent under concept shift, because the change is in the very relationship the labels reveal; this bounds what any decision-time monitor can promise. A practical diagnostic follows: E1 and E3 agree under covariate shift with a well-specified model and diverge when calibration breaks, so their disagreement is itself a warning signal (Section VI-G). An alert is raised for period *T* when the estimate exceeds δ_{R} in absolute value; we use δ_{R} = 2 pp.", { noIndent: true }));
add(tableCaption("II", "Conditions Under Which Each Residual Estimator Is Unbiased"));
add(table(["Drift", "Stable", "E0", "E1", "E2", "E3"],
  [["None", "all", "yes", "if calibrated", "yes", "yes"],
   ["Covariate", "P(y | x, g)", "lags", "if well specified", "approx.", "yes"],
   ["Label", "P(x | y, g)", "lags", "no guarantee", "no guarantee", "no guarantee"],
   ["Concept", "P(x, g)", "lags", "no guarantee", "no guarantee", "no guarantee"]],
  [760, 900, 560, 980, 920, 920], { size: 14 }));
add(tableNote("“Lags”: exact only if nothing changed in the last *d* periods."));

// =========================================================== V. EXPERIMENTAL SETUP
add(h1("V. Experimental Setup"));
add(h2("A. Data and Tasks"));
add(body("We use the American Community Survey (ACS) one-year Public Use Microdata Samples through the Folktables interface [18] for 2014–2019 and 2021–2023; no standard one-year file was released for 2020. We use two tasks with the standard Folktables filters and features: ACSIncome (favourable outcome: income above a threshold) and ACSEmployment (favourable outcome: employed). Groups are defined by race (Black = 1, White = 0; other races excluded) and by sex (female = 1, male = 0). The main state is California; Texas, New York and Florida are pooled as an out-of-state replication. The group attribute is excluded from the model inputs. Occupation and place of birth are coarsened to their leading digits, and the 2019 replacement of the relationship variable RELP by RELSHIPP is mapped back to the earlier coding.", { noIndent: true }));
add(body("Because the nominal $50,000 threshold of ACSIncome is not adjusted for inflation, we use three income labels: nominal ($50,000 in current dollars), real ($50,000 in 2015 dollars, using the ACS adjustment factor and the CPI-U annual average for the U.S. city average, all items [37]) and quantile (the top 40% of real income within each year), which removes level drift entirely. The quantile label is the default. Table III lists the settings; the corporate credit setting is described in Section VII."));

section(1);
add(tableCaption("III", "Experimental Settings"));
add(table(T.settings.header, T.settings.rows, [1500, 1530, 1350, 1250, 300, 1050, 1050, 2410], { size: 14 }));
add(tableNote("Favourable rates are the minimum and maximum over test years. Individuals per test year are counts after the Folktables filters and the restriction to the two groups."));
section(2);

add(h2("B. Protocol"));
add(body("For each test year *T*, a model is trained on all years up to *T* − *d* (expanding window; *d* = 1 in the main analysis and *d* = 2 as a robustness check, which matches the corporate credit design), with at most 400,000 training rows sampled at random. Individuals of year *T* are scored and the top *q* share is approved, with *q* = 0.75 in the main text (results for *q* ∈ {0.60, 0.90} are similar). The tolerance is δ = 5 pp, which corresponds to an adverse-impact ratio of about 0.93 at this approval rate and is stricter than the four-fifths rule [32]; Table A1 reports counts for δ from 1 to 15 pp. β^{*}_{t} is searched on a grid of 1001 values.", { noIndent: true }));
add(body("For the delay-aware estimators, the latest labelled year *H* is excluded from training and used only as the reference set. Approval-gap policies are compared under four rules: no calibration; static calibration (β^{*} of the first test year, then frozen); carry-over (the previous year’s β^{*}); and per-period calibration (β^{*}_{T}, computable at decision time and therefore an upper benchmark for the approval gap)."));
add(h2("C. Models"));
add(body("We use logistic regression (one-hot encoded categorical features, L2 penalty, *C* = 1), XGBoost [30] (300 trees, learning rate 0.05, depth 5, row and column subsampling 0.8) and LightGBM [31] (300 trees, learning rate 0.05, 31 leaves, subsampling 0.8). Hyperparameters were fixed before the analysis and not tuned per year. Out-of-time AUC is 0.86–0.89 for logistic regression and 0.87–0.91 for the tree ensembles across the census settings.", { noIndent: true }));
add(h2("D. Inference"));
add(body("Yearly intervals come from a nonparametric bootstrap with 200 replicates that resamples individuals within the year and holds the model fixed. Pooled intervals resample within every year and aggregate. Intervals therefore reflect sampling variation in the evaluation data, not retraining variation [25].", { noIndent: true }));
add(h2("E. Synthetic Benchmark"));
add(body("To validate the decomposition and the estimators against known ground truth, we simulate nine periods with five Gaussian features, a 20% minority group and 20,000 individuals per period under four scenarios: no drift; covariate shift (the feature means of group 1 fall by 0.8 standard deviations over the horizon with *P*(*y* | *x*) fixed); label shift (the favourable rate of group 1 falls from 40% to 20% with *P*(*x* | *y*) fixed); and concept shift (the intercept of *P*(*y* | *x*, *g* = 1) falls by 1.2 on the logit scale with *P*(*x*) fixed). A group-blind logistic regression is retrained each period with *d* = 1, and each scenario is replicated 20 times.", { noIndent: true }));
add(h2("F. Implementation"));
add(body("All experiments use Python 3.14.0 with scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.7.0, Folktables 0.0.12, pandas 3.0.2, NumPy 2.4.4 and Matplotlib 3.10.9 on an Apple Silicon laptop. A full audit of one California setting with three models takes 130–220 s, the three-state setting 384 s, and one run of the delay-aware monitor 60–100 s.", { noIndent: true }));

// =========================================================== VI. RESULTS
add(h1("VI. Results"));
add(body("Unless stated otherwise, results refer to XGBoost at *q* = 0.75 with the quantile income label and *d* = 1; logistic regression and LightGBM give the same qualitative picture (Table IV).", { noIndent: true }));
add(h2("A. Main Case: A Level Break Hidden by Pooling"));
add(body("In California income prediction, the approval gap between Black and White individuals was stable before the pandemic and roughly tripled after it (Fig. 2b). From 2015 to 2019 the yearly gap ranged from −2.7 to −4.9 pp, inside the ±5 pp tolerance; from 2021 to 2023 it ranged from −11.0 to −12.2 pp, with all bootstrap intervals excluding zero. The pooled gap was −6.4 pp (95% CI −6.8 to −6.0), a value that lies between the two regimes and describes neither. Over the same period the favourable rate of White individuals rose from 41.8%–43.4% in 2015–2019 to 48.7%–49.7% in 2021–2023, while that of Black individuals stayed between 31.9% and 34.6%.", { noIndent: true }));

section(1);
add(figure("fig2_yearly_gaps.png", 7.0, 2, "Yearly approval gaps (group 1 minus group 0) and residual gaps among qualified individuals, XGBoost, *q* = 0.75. Panels (a)–(e): the five drifting settings; the dotted line marks the pooled gap. Panel (f): California income × race under three income-label definitions and two label delays."));
section(2);

add(h2("B. A Taxonomy of Audit Failures"));
add(body("Table IV summarizes all settings. Each displays a distinct temporal pattern, and each pattern defeats a pooled audit in a different way.", { noIndent: true }));
add(item("•", "**Cancellation.** In the corporate credit case, yearly gaps changed sign and offset each other: the pooled gap of −0.8 pp is about one tenth of the mean absolute yearly gap of 8.7 pp (Section VII)."),
  item("•", "**Level break.** In California income, the pooled gap lies between two regimes, and the pooled calibration strength fits neither (Section VI-D)."),
  item("•", "**High level with a step.** In the three other states, the gap was large throughout (−10.5 to −11.4 pp in 2015–2019) and stepped to −14.6 pp in 2021. A pooled audit detects the problem here, but the required calibration strength still moved from 0.55–0.60 to 0.66–0.70."),
  item("•", "**Gradual drift with a peak.** For employment, the gap widened from −3.6 pp in 2016 to −5.8 pp in 2019, peaked at −8.0 pp in 2021 and partly recovered to −6.3 pp in 2023."),
  item("•", "**Convergence.** For income by sex, the gap narrowed from about −6.5 pp in 2015–2019 to −4.8 pp in 2023, and β^{*}_{t} fell from 0.22–0.23 to 0."),
  item("•", "**Stable control.** For employment by sex, the gap stayed between +1.1 and +1.7 pp under XGBoost (+1.9 to +2.5 pp under logistic regression), the pooled gap of +1.4 pp represented every year, and no calibration was needed in any year under any model. The audit does not manufacture instability where there is none."));

section(1);
add(tableCaption("IV", "Pooled Versus Period-Wise Audit Statistics (q = 0.75)"));
add(table(T.summary.header, T.summary.rows, [1650, 1150, 900, 1600, 900, 950, 1000, 700, 1590],
  { size: 14, leftCols: 3, ruleAfter: [2, 5, 8, 11, 14] }));
add(tableNote("Negative gaps: group 1 is approved less often. Corporate credit: test years 2014–2022, *d* = 2, from [TBD: cite the earlier working paper if desired]. Carry-over applies the previous year’s β^{*}; the number in parentheses is the largest absolute gap in a year outside the tolerance."));
section(2);

add(h2("C. Which Component Moves"));
add(body("Proposition 2 separates the patterns further (Fig. 2, dashed lines). In California income, the risk-composition component accounted for 73% of the post-break gap, but the residual also widened, from −0.4 to −0.8 pp in 2015–2018 to between −2.8 and −3.3 pp in 2021–2023: after the break, qualified Black individuals were approved less often than qualified White individuals. In the three other states the residual was −2.1 to −2.7 pp before 2021 and −2.9 to −4.0 pp after. For employment by race, by contrast, the residual never exceeded 0.3 pp in absolute value, so the gap arose entirely among individuals who were not employed. In the sex-based settings the residual was essentially zero.", { noIndent: true }));
add(body("The employment case also shows why the decomposition is needed rather than a comparison of base rates. The difference in employment rates between the groups was nearly constant (−6.8 pp in 2015 and 2019, −7.1 pp in 2021, −5.4 pp in 2023), yet the approval gap widened from −4.2 to −8.0 pp; across years the two were essentially uncorrelated (*r* = −0.14). A change in the gap is therefore not evidence of a change in relative risk, and vice versa."));
add(h2("D. Calibration Parameters Do Not Transfer"));
add(body("The pooled calibration strength fits no individual year in the drifting settings (Fig. 3). In California income, β^{*}_{pool} = 0.25, whereas β^{*}_{t} was 0 in 2015–2019 and 0.59–0.63 in 2021–2023: applying the pooled value over-corrects every pre-break year and supplies less than half the required correction after the break. In the convergence setting, β^{*}_{pool} = 0.16 under-corrects 2015–2019 (0.22–0.23 needed) and over-corrects 2022–2023 (0–0.02 needed).", { noIndent: true }));
add(body("Carrying over the previous year’s β^{*} left the gap outside the tolerance in 3 of 7 years for California income, the three-state setting and employment by race, with gaps of up to 11.2, 6.7 and 7.1 pp, respectively. In all three settings the largest exceedance fell in 2021, the first year after the break; the remaining failures (2017–2019 and 2022–2023) exceeded the band by 0.1–0.5 pp. For income by sex, carry-over failed in 1 of 7 years, by only 0.06 pp; we therefore report the size of each exceedance and not only the count. Table A1 shows that the conclusions do not hinge on the 5-pp tolerance: in California income, three years exceed any band from 5 to 10 pp."));

section(1);
add(figure("fig3_beta_mismatch.png", 7.0, 3, "Calibration strength required in each year (β^{*}_{t}, points) versus the strength set once on the pooled sample (β^{*}_{pool}, dashed), XGBoost, *q* = 0.75, δ = 5 pp."));
section(2);

add(h2("E. Robustness"));
add(body("The break in California income is not an artefact of the label definition or the label delay (Table V, Fig. 2f). The quantile label fixes the overall favourable rate in every year and thus removes inflation and wage growth entirely; the break persists. Lengthening the delay from one to two years leaves it essentially unchanged, so the break is not a matter of how stale the training data are but a shift that no model trained on earlier years anticipates. Logistic regression produced larger gaps in the race settings (up to −13.9 pp in California) and smaller gaps in the income-by-sex setting (−3.0 to −4.9 pp, inside the tolerance in every year), so the choice of model alone can change an audit’s verdict (Table IV). Two further checks address the census data themselves (last three rows of Table V). With ACS person weights in the label quantile, model training, thresholds and approval rates, the pre-break gap almost vanished (−0.4 to −1.6 pp) but the gap still jumped to −8.6 to −9.3 pp in 2021–2023; weighting changes the level of the gap, not the break. Removing occupation, whose coding changed in 2018, left the break intact (−2.8 to −9.9 pp on average). Under every variant, exactly the three post-break years were outside the tolerance, and the post-break residual stayed between −2.0 and −3.0 pp.", { noIndent: true }));
section(1);
add(tableCaption("V", "Robustness of the California Break (Income × Race, XGBoost)"));
add(table(T.robust.header, T.robust.rows, [3300, 500, 1300, 1300, 1200, 1300, 1000], { size: 14 }));
add(tableNote("Mean gaps are averages of the yearly gaps; for *d* = 2 the first test year is 2016. The last three rows come from a separate check in which the 40% income threshold is computed among Black and White individuals only; compare them with their own baseline row. Their pooled gaps omit the composition term of Proposition 1, and β^{*}_{pool} was not computed."));

section(2);

add(h2("F. Synthetic Validation"));
add(body("The synthetic benchmark confirms which component each shift moves and when each estimator works (Table VI). First, label shift moves only the risk-composition component, as Proposition 2 implies, whereas covariate shift also moves the residual, because qualified members of group 1 move closer to the threshold; a residual is relative to the label, not a measure of unequal treatment. Second, under covariate shift the last observed value E0 lags with a bias of 1.74 pp, and E3 removes it (0.11 pp), close to E3 with the true density ratio (0.04 pp). Third, under concept shift E1 is the most biased (−1.99 pp) because the model’s calibration breaks, and no label-free estimator tracks the change reliably. Mean absolute errors include sampling noise of about 1 pp per period (under covariate shift: E0 2.11, E1 1.10, E2 1.49, E3 1.54 pp). The same numbers were obtained on two machines.", { noIndent: true }));

section(1);
add(tableCaption("VI", "Synthetic Benchmark With Known Ground Truth (20 Replicates)"));
add(table(T.synth.header, T.synth.rows, [1500, 1350, 1350, 1350, 900, 900, 900, 900, 1290], { size: 14 }));
add(tableNote("First and last test periods are 2 and 8. Bias = estimate minus realized *RG*_{t}, averaged over periods 2–8, in pp. “True ratio” uses the known density ratio in place of the estimated one."));
section(2);

add(h2("G. Monitoring at Decision Time"));
add(body("At the California break, the proposed estimator recovered most of the residual before labels arrived, whereas the last observed value recovered less than half (Fig. 4, Table VII). The realized residual in 2021 was −3.06 pp. E3 estimated −2.69 pp (88% of the realized value), E2 −2.59 pp (85%), E1 −2.00 pp (65%) and E0 −1.40 pp (46%). These results use XGBoost, the quantile label and a logistic domain classifier; because the reference year is held out of training, test years start in 2016 and realized residuals differ slightly from Section VI-A.", { noIndent: true }));
add(body("E2 and E3 performed similarly overall: E2 had the lowest mean error, E3 the lowest maximum error and the smallest error at the break. Both detected all three years in which the residual exceeded 2 pp, without false alerts, while E0 and E1 each missed one. The weights were benign: the effective sample size stayed above 99% of the reference set except in 2021, when it fell to 92% (89% with the boosted classifier), itself a sign that the covariate distribution moved. The choice of domain classifier mattered little: with boosted trees, E3 estimated −2.50 pp in 2021 (82%)."));
add(...figure("fig4_monitoring.png", 3.4, 4, "Realized residual gap (after labels arrive) and its four decision-time estimates, California income × race, *d* = 1, logistic domain classifier."));
add(body("E1 diverged from E2 and E3 after the break. From 2021 it stayed near −2.0 to −2.2 pp while the realized residual was −2.8 to −3.1 pp, an underestimate of about one point. This is the pattern Section IV-D predicts when the model’s calibration breaks, and the E1–E3 gap of 0.7–1.2 pp from 2021 onwards would have flagged it at decision time."));
add(body("A longer label delay widens the advantage of the reweighted estimators. With *d* = 2, the mean absolute error of E0 rose from 0.57 to 0.83 pp and it detected only one of the three years beyond 2 pp, because the latest labelled year now lags by two periods; for 2021 and 2022 both reference sets were 2019, since no 2020 file exists. E2 and E3 were nearly unaffected (0.36 and 0.45 pp) and again detected all three years without false alerts; in 2022, E3 estimated −2.66 pp against a realized −3.02 pp, while E0 still reported −1.40 pp."));
add(body("The employment setting serves as a negative control for the monitor. Its realized residual stayed between −0.29 and +0.05 pp in every year; all estimators tracked it with mean absolute errors of 0.07–0.12 pp and maximum errors of at most 0.20 pp, and none raised an alert. The monitor stays quiet where there is no residual to report, even in 2021, when the effective sample size of the weights fell to about 90%."));
add(body("One limitation is shared by all estimators. In 2019, the first year in which the residual widened (−1.28 to −1.54 pp, depending on *d*), every estimator stayed between −0.35 and −1.20 pp. The onset of a shift in *P*(*y* | *x*) cannot be seen without its labels; the estimators become accurate once the covariate distribution itself has moved, as it did in 2021."));
add(tableCaption("VII", "Error of Decision-Time Residual Estimates (β = 0)"));
add(table(T.monitor.header, T.monitor.rows, [1180, 560, 640, 640, 620, 700, 700], { size: 14 }));
add(tableNote("XGBoost, quantile label where applicable. E3 uses a logistic (default) or gradient-boosted domain classifier. Alerts at δ_{R} = 2 pp."));
add(body("Table VIII compares approval-gap policies. Because β^{*} was 0 in 2016, the static policy coincided with no calibration in California. Per-period calibration met the tolerance in every year at no visible cost in false rejections of qualified individuals, and in both settings it also kept the residual below 2 pp. Whether that holds in general is an empirical question; the decision-time estimators are what allow an auditor to check it before labels arrive."));
add(tableCaption("VIII", "Approval-Gap Policies Under the Monitoring Protocol"));
add(table(T.policy.header, T.policy.rows, [1500, 1180, 1180, 1180], { size: 14 }));
add(tableNote("XGBoost, *q* = 0.75, test years 2016–2023. Static: β^{*} of the first test year, frozen. The false-rejection rate is the share of qualified individuals not approved."));

// =========================================================== VII. CASE STUDY
add(h1("VII. Case Study: Corporate Credit Screening"));
add(body("The census tasks have no natural label delay and no sign reversal; the corporate credit case supplies both. Using annual statements of 4,974 non-financial Chinese listed firms from the CSMAR database, including firms later delisted, we simulate a credit screen that approves the 75% of firms with the lowest predicted two-year financial-distress risk. Each test year *T* from 2014 to 2022 uses a model trained on years up to *T* − 2, because two-year outcomes are observed only then (*d* = 2). The 16 predictors are financial ratios; groups are heavy-polluting industries (*g* = 1) and all other industries, a grouping whose relative risk follows environmental policy and the industrial cycle [34]. The test set has 32,497 firm-years, 7,528 of them heavy-polluting.", { noIndent: true }));
add(body("**Cancellation.** Under XGBoost, the pooled approval gap was −0.8 pp (95% CI −2.7 to 1.2), statistically indistinguishable from zero. Yet yearly gaps ranged from −16.5 pp in 2015 to +8.1 pp in 2021, with a mean absolute value of 8.7 pp (Fig. 2a); the composition term of Proposition 1 was 0.18 pp, so the pooled gap is, almost exactly, an average of offsetting yearly gaps. Six of nine years lay outside ±5 pp, β^{*}_{t} ranged from 0 to 0.76, β^{*}_{pool} was 0, and carrying over the previous year’s β^{*} failed in 3 of 8 years."));
add(body("**A binding bound.** Distress was rare in 2014–2016 (2.7%–6.1% in both groups), so Proposition 2 bounds the risk-composition component tightly: it was at most 1.3 pp in absolute value in 2014–2018. The gaps of −12.7 to −16.5 pp in 2014–2016 were therefore almost entirely residual (−12.3 to −15.7 pp among firms that did not become distressed), and remained so when distress was measured over four years (−11.9 to −15.1 pp). After 2017 heavy-polluting firms became relatively safer, and the gap reversed sign. Across years the gap correlated at −0.91 with the difference in realized distress rates."));
add(body("The case complements the census results. It shows the cancellation mode, a setting where the bound of Proposition 2 is informative, and a residual that dominates the gap, whereas in the census tasks the risk-composition component dominates."));

// =========================================================== VIII. DISCUSSION
add(h1("VIII. Discussion"));
add(h2("A. Implications for Auditing Retrained Models"));
add(body("Four practices follow. First, audits should be run per period, or per decision batch, and report the dispersion of period gaps and the number and size of exceedances next to any pooled figure; a small pooled gap is a statement about an average, not about any period. Second, each period gap should be decomposed into its risk-composition and residual components, because gaps of similar size can have different origins: risk composition in the census race settings, residual in early corporate credit years. Third, calibration strength should be recomputed for every decision batch. Because the approval gap needs no labels, this is always feasible and, by construction, meets the tolerance in every period where some β does; static and carry-over policies do not (Table VIII). Fourth, the residual, which per-batch parity calibration does not target, should be monitored with decision-time estimates and confirmed once labels arrive.", { noIndent: true }));
add(body("The stable control matters as much as the failures. Where group risk profiles do not drift, pooled and period-wise audits agree and the period-wise procedure costs little; the framework reveals instability where it exists rather than creating it."));
add(h2("B. Limitations"));
add(item("1)", "**No feedback.** Decisions do not affect future populations in our design. This isolates measurement but excludes the dynamics studied in [11], [12]."),
  item("2)", "**Census data.** The main analysis is unweighted; survey weights lower the level of the gap but not the break (Section VI-E). The ACSIncome population is restricted to people with earnings, whose composition changed during the pandemic; and no standard 2020 file exists, so the break cannot be dated within 2019–2021. Occupation codes were revised in 2018; excluding occupation leaves the break intact."),
  item("3)", "**Groups.** Race is coded as Black versus White, excluding other groups, and sex as binary, as recorded in the ACS."),
  item("4)", "**Residuals are relative to labels.** A residual measures unequal approval among people with the same observed outcome. It moves under pure covariate shift (Section VI-F) and is not a measure of discrimination, which concerns the decision process and protected grounds."),
  item("5)", "**Label-free estimation has limits.** Under concept shift no estimator that ignores current labels is reliable, and the onset of a shift is invisible (Section VI-G); decision-time estimates are warnings, not verdicts."),
  item("6)", "**Inference.** Bootstrap intervals hold models fixed and exclude retraining variation. With eight or nine periods per setting, cross-year associations are descriptive."),
  item("7)", "**Corporate credit.** The screen is simulated; financial distress is a statement-based proxy, not a loan default, and the industry grouping is not a protected class."));

// =========================================================== IX. CONCLUSION
add(h1("IX. Conclusion"));
add(body("Post-processing fairness parameters are often set once, on pooled data, for models that are retrained period after period. We showed with two exact identities that pooled audits can fail in distinct ways (cancellation, level break, gradual drift and convergence) and that each period gap separates into a risk-composition part and a residual among qualified individuals. Across census income and employment tasks, race and sex groups, four states and a corporate credit case, these failure modes appeared under every label definition, label delay and model class we tried, while a stable setting confirmed that the procedure does not invent instability. Because the residual depends on delayed labels, we proposed a decision-time estimator that reweights the latest labelled period towards current covariates. It removes the lag of the last observed value under covariate shift and, in California, detected every year in which the residual exceeded 2 pp without false alerts; no label-free estimator can be trusted under concept shift. Fairness audits of retrained classifiers should therefore be conducted per period, report both components, recompute calibration for every batch, and monitor the residual before labels arrive.", { noIndent: true }));

// =========================================================== BACK MATTER
add(h1("Acknowledgment"));
add(body("The author used Claude (Anthropic, [TBD: model version]) during this study to discuss the research design, to write the data-processing, audit and monitoring code, to produce the figures and tables, and to draft the first version of all sections of this article. The author re-ran all experiments, checked every reported number against the output files, verified the cited sources and revised the text [TBD: edit to describe exactly what you did]. The author takes full responsibility for the content.", { noIndent: true }));
add(h1("Code and Data Availability"));
add(body("The census data are public and are downloaded by the code through Folktables. The CSMAR data are licensed and cannot be redistributed; the processing code is provided. [TBD: repository link with code, configuration files and seeds.]", { noIndent: true }));

add(h1("References"));
const REFS = [
  "M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2016, pp. 3315–3323.",
  "N. Kozodoi, J. Jacob, and S. Lessmann, “Fairness in credit scoring: Assessment, implementation and profit implications,” *Eur. J. Oper. Res.*, vol. 297, no. 3, pp. 1083–1094, 2022.",
  "S. Barocas and A. D. Selbst, “Big data’s disparate impact,” *Calif. Law Rev.*, vol. 104, no. 3, pp. 671–732, 2016.",
  "European Union, “Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act),” *Off. J. Eur. Union*, Jul. 2024.",
  "A. Fuster, P. Goldsmith-Pinkham, T. Ramadorai, and A. Walther, “Predictably unequal? The effects of machine learning on credit markets,” *J. Finance*, vol. 77, no. 1, pp. 5–47, 2022.",
  "S. Das, R. Stanton, and N. Wallace, “Algorithmic fairness,” *Annu. Rev. Financ. Econ.*, vol. 15, pp. 565–593, 2023.",
  "J. Kleinberg, S. Mullainathan, and M. Raghavan, “Inherent trade-offs in the fair determination of risk scores,” in *Proc. Innov. Theor. Comput. Sci. Conf. (ITCS)*, 2017, pp. 43:1–43:23.",
  "A. Chouldechova, “Fair prediction with disparate impact: A study of bias in recidivism prediction instruments,” *Big Data*, vol. 5, no. 2, pp. 153–163, 2017.",
  "G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and K. Q. Weinberger, “On fairness and calibration,” in *Proc. NeurIPS*, 2017, pp. 5680–5689.",
  "S. Corbett-Davies and S. Goel, “The measure and mismeasure of fairness: A critical review of fair machine learning,” 2018, *arXiv:1808.00023*.",
  "L. T. Liu, S. Dean, E. Rolf, M. Simchowitz, and M. Hardt, “Delayed impact of fair machine learning,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, pp. 3150–3158.",
  "A. D’Amour, H. Srinivasan, J. Atwood, P. Baljekar, D. Sculley, and Y. Halpern, “Fairness is not static: Deeper understanding of long term fairness via simulation studies,” in *Proc. ACM Conf. Fairness, Accountability, Transparency (FAccT)*, 2020, pp. 525–534.",
  "A. Barrainkua, P. Gordaliza, J. A. Lozano, and N. Quadrianto, “Preserving the fairness guarantees of classifiers in changing environments: A survey,” *ACM Comput. Surv.*, vol. 57, no. 6, pp. 1–32, 2025.",
  "H. Singh, R. Singh, V. Mhasawade, and R. Chunara, “Fairness violations and mitigation under covariate shift,” in *Proc. ACM FAccT*, 2021. [VERIFY pages]",
  "A. Rezaei, A. Liu, O. Memarrast, and B. D. Ziebart, “Robust fairness under covariate shift,” in *Proc. AAAI Conf. Artif. Intell.*, 2021. [VERIFY pages]",
  "S. Maity, D. Mukherjee, M. Yurochkin, and Y. Sun, “Does enforcing fairness mitigate biases caused by subpopulation shift?” in *Proc. NeurIPS*, 2021. [VERIFY pages]",
  "J. Schrouff *et al.*, “Diagnosing failures of fairness transfer across distribution shift in real-world medical settings,” in *Proc. NeurIPS*, 2022. [VERIFY pages]",
  "F. Ding, M. Hardt, J. Miller, and L. Schmidt, “Retiring Adult: New datasets for fair machine learning,” in *Proc. NeurIPS*, 2021. [VERIFY pages]",
  "H. Shimodaira, “Improving predictive inference under covariate shift by weighting the log-likelihood function,” *J. Stat. Plan. Inference*, vol. 90, no. 2, pp. 227–244, 2000.",
  "M. Sugiyama, M. Krauledat, and K.-R. Müller, “Covariate shift adaptation by importance weighted cross validation,” *J. Mach. Learn. Res.*, vol. 8, pp. 985–1005, 2007.",
  "Z. Lipton, Y.-X. Wang, and A. Smola, “Detecting and correcting for label shift with black box predictors,” in *Proc. ICML*, 2018. [VERIFY pages]",
  "S. Rabanser, S. Günnemann, and Z. Lipton, “Failing loudly: An empirical study of methods for detecting dataset shift,” in *Proc. NeurIPS*, 2019. [VERIFY pages]",
  "S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, and H. Sedghi, “Leveraging unlabeled data to predict out-of-distribution performance,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.",
  "J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, “A survey on concept drift adaptation,” *ACM Comput. Surv.*, vol. 46, no. 4, Art. no. 44, 2014.",
  "K. Lum, Y. Zhang, and A. Bower, “De-biasing ‘bias’ measurement,” in *Proc. ACM FAccT*, 2022, pp. 379–389.",
  "C. Hurlin, C. Pérignon, and S. Saurin, “The fairness of credit scoring models,” *Manage. Sci.*, vol. 72, no. 1, pp. 406–425, 2026.",
  "B. Zadrozny and C. Elkan, “Transforming classifier scores into accurate multiclass probability estimates,” in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining*, 2002, pp. 694–699.",
  "A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in *Proc. ICML*, 2005, pp. 625–632.",
  "S. John, “Fair and explainable credit-scoring under concept drift: Adaptive explanation frameworks for evolving populations,” 2025, *arXiv:2511.03807*.",
  "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. ACM SIGKDD*, 2016, pp. 785–794.",
  "G. Ke *et al.*, “LightGBM: A highly efficient gradient boosting decision tree,” in *Proc. NeurIPS*, 2017. [VERIFY pages]",
  "U.S. Equal Employment Opportunity Commission *et al.*, “Uniform guidelines on employee selection procedures,” 29 C.F.R. Part 1607, 1978.",
  "Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency, “Supervisory guidance on model risk management,” SR Letter 11-7, 2011.",
  "Q. Qiu and J. Yu, “Green credit policy and default risk of the heavy polluting corporations,” *J. Cleaner Prod.*, vol. 455, Art. no. 142291, 2024.",
  "J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf, and A. Smola, “Correcting sample selection bias by unlabeled data,” in *Proc. NeurIPS*, 2006. [VERIFY pages]",
  "S. Bickel, M. Brückner, and T. Scheffer, “Discriminative learning under covariate shift,” *J. Mach. Learn. Res.*, vol. 10, pp. 2137–2155, 2009.",
  "U.S. Bureau of Labor Statistics, \u201cConsumer Price Index for All Urban Consumers (CPI-U): U.S. city average, all items, 1982\u201384 = 100 (series CUUR0000SA0), annual averages,\u201d 2026. [Online]. Available: https://www.bls.gov/cpi/ (accessed Oct. 27, 2026).",
];
add(new Paragraph({ spacing: { after: 60 }, children: runs("[VERIFY every reference against the original publication before submission.]", { size: 16 }) }));
REFS.forEach((r, i) => add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
  indent: { left: 360, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 }],
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ children: [new Tab()], size: 16 }), ...runs(r, { size: 16 })] })));

// ---- appendix
add(h1("Appendix"));
add(body("Table A1 counts the test years in which the absolute yearly gap exceeds tolerances from 1 to 15 pp. In every drifting setting the count changes gradually with the tolerance, so no conclusion depends on the choice of 5 pp.", { noIndent: true }));
section(1);
add(tableCaption("A1", "Years Outside the Tolerance Band for Different Tolerances (XGBoost, q = 0.75)"));
add(table(T.delta.header, T.delta.rows, [2240, 820, 820, 820, 820, 820, 820, 820, 820, 820, 820], { size: 14 }));
add(tableNote("Columns give the tolerance δ in pp; cells give years with |*G*_{t}| > δ out of all test years."));
section(2);

// ---- biographies
add(new Paragraph({ spacing: { before: 240, after: 60 }, children: [
  new TextRun({ text: "[Photo 1 in × 1.25 in]  ", font: SANS, size: 16, highlight: "yellow" }),
  new TextRun({ text: "WENRUI QIANG ", font: SANS, size: 16, bold: true }),
  ...runs("[TBD: short biography, required for all authors by IEEE Access: degrees, current position, research interests.]", { size: 16 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
  children: [new TextRun({ text: "■", font: SANS, size: 20, color: BLUE })] }));

// =========================================================== ASSEMBLE
const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  children: [new TextRun({ text: "W. Qiang: Period-Wise Decomposition and Delay-Aware Monitoring of Group Approval Gaps", font: SANS, size: 14, color: "666666" })] })] });
const footer = new Footer({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: FULL_W }],
  children: [new TextRun({ text: "VOLUME XX, 2026", font: SANS, size: 14, color: "666666" }),
    new TextRun({ children: [new Tab(), PageNumber.CURRENT], font: SANS, size: 14, color: "666666" })] })] });
const page = { size: { width: PAGE_W, height: 15840 }, margin: { top: 1080, bottom: 1080, left: MARGIN_LR, right: MARGIN_LR, header: 540, footer: 540 } };
const doc = new d.Document({
  creator: "Wenrui Qiang", title: "When Pooled Fairness Audits Fail",
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SANS, size: 19, bold: true, color: BLUE }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SANS, size: 19, italics: true, color: BLUE }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 1 } }] },
  sections: sections.map((s) => ({
    properties: { page, type: s.first ? undefined : SectionType.CONTINUOUS,
      column: s.cols === 2 ? { count: 2, space: COL_GAP, equalWidth: true } : { count: 1 } },
    headers: s.first ? { default: header } : undefined, footers: s.first ? { default: footer } : undefined,
    children: s.children.length ? s.children : [new Paragraph({ children: [] })] })),
});
d.Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("manuscript_IEEE_Access_draft.docx", buf);
  console.log("written", buf.length, "bytes;", sections.length, "sections");
});
