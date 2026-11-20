"""Write the supplemental decomposition table from the historical audit rates."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
records = ROOT / "results" / "matched" / "california_components.csv"
target = ROOT / "paper" / "generated" / "components.tex"
columns = ("pi1", "pi0", "a11", "a01", "a10", "a00", "G", "RG", "RC")
with records.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if [int(r["year"]) for r in rows] != [2015,2016,2017,2018,2019,2021,2022,2023]:
    raise ValueError("Expected the eight original audit cohorts")
lines = [
    r"\begin{table*}[!t]",
    r"\centering\caption{Annual Outcome and Approval Rates Underlying the California Income Gap}\label{tab:components}",
    r"\footnotesize\begin{tabular}{rlrrrrrrrrr}\toprule",
    r"Year & $y=1$ counts ($g=1/0$) & $\pi_1$ & $\pi_0$ & $a_{11}$ & $a_{01}$ & $a_{10}$ & $a_{00}$ & $G_t$ & $RG_t$ & $RC_t$ \\ \midrule",
]
for row in rows:
    p={k:float(row[k]) for k in columns}
    if not all(map(math.isfinite,p.values())) or not math.isclose(p["G"],p["RG"]+p["RC"],abs_tol=1e-10):
        raise ValueError("Incomplete or inconsistent conditional-rate data")
    group_n1=int(row["n1"])
    group_n0=int(row["n"])-group_n1
    qualified=[p["pi1"]*group_n1,p["pi0"]*group_n0]
    if any(not math.isclose(value,round(value),abs_tol=1e-6) or value<1 for value in qualified):
        raise ValueError("Cannot reconstruct the observed favourable-outcome counts")
    counts=" / ".join(f"{round(value):,}" for value in qualified)
    values = [f"{100 * p[col]:.2f}" for col in columns]
    lines.append(" & ".join([row["year"],counts, *values]) + r" \\")
lines.extend([
    r"\bottomrule\end{tabular}",
    r"\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize The historical California ACS income audit used XGBoost with $q=.75$ and $\beta=0$. The two counts give the denominators for $a_{11}$ (Black respondents) and $a_{01}$ (White respondents). Within group $g$, $\pi_g$ is the prevalence of favourable outcomes and $a_{gy}$ is the approval rate for outcome $y$. Rates are percentages. $G_t$ is the overall Black minus White approval gap, $RG_t$ is the same difference among respondents with favourable outcomes and $RC_t=G_t-RG_t$. These gaps are in percentage points, calculated before rounding.\end{minipage}",
    r"\end{table*}",
])
target.parent.mkdir(exist_ok=True, parents=True)
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
