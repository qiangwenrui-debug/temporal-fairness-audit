"""Reweight the current ACS score distribution to the preceding cohort's observed mix."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
YEARS = [*range(2011, 2020), *range(2021, 2025)]
STATES = ("CA", "TX", "NY", "FL")
SCENARIOS = ("observed", "prior_group_mix", "prior_within_group", "prior_joint")
POLICIES = ("pooled", "previous", "common")
N_CELLS = 24


@dataclass
class Cohort:
    score: np.ndarray
    groups: np.ndarray
    cells: np.ndarray
    households: np.ndarray
    orders: list
    summaries: list
    cell_counts: np.ndarray

    @property
    def n(self):
        return len(self.score)


def read_cohort(root, state, year):
    data = root / "data"
    extended = data / "early_rolling"
    directory = extended if year < 2014 else data
    fields = ["AGEP", "SEX", "SCHL", "WKHP", "PINCP", "PWGTP", "RAC1P", "ADJINC", "SERIALNO"]
    raw = pd.read_parquet(directory / f"normalized/{state}_{year}.parquet", columns=fields)
    working = raw[(raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)]
    adjusted_income = working.PINCP * working.ADJINC
    outcome = adjusted_income > adjusted_income.quantile(.60)
    keep = working.RAC1P.isin([1, 2])
    people = working[keep]
    if people[fields].isna().any().any():
        raise ValueError(f"Missing covariate or identifier for {state} {year}")
    group = (people.RAC1P.to_numpy() == 2).astype(int)
    task = f"{state}_income_race_quantile_d1_s0"
    with np.load(extended / f"scores/audit/{task}/{year}.npz") as saved:
        np.testing.assert_array_equal(group, saved["g"])
        np.testing.assert_array_equal(outcome[keep].to_numpy(), saved["y"])
        score = saved["s"].astype(float)
    age = people.AGEP.to_numpy()
    female = people.SEX.to_numpy() == 2
    bachelor = people.SCHL.to_numpy() >= 21
    full_time = people.WKHP.to_numpy() >= 35
    age_band = np.searchsorted([35, 55], age, side="right")
    cells = group*N_CELLS + age_band*8 + female*4 + bachelor*2 + full_time
    counts = np.bincount(cells, minlength=2*N_CELLS)
    summaries = []
    for g, name in enumerate(("White", "Black")):
        subset = group == g
        summaries.append(dict(state=state, year=year, group=name, n=int(subset.sum()),
                              group_share=subset.mean(), mean_age=age[subset].mean(),
                              age_under35_share=(age_band[subset] == 0).mean(),
                              age_55plus_share=(age_band[subset] == 2).mean(),
                              female_share=female[subset].mean(),
                              bachelor_share=bachelor[subset].mean(),
                              full_time_share=full_time[subset].mean(),
                              mean_weekly_hours=people.WKHP.to_numpy()[subset].mean(),
                              min_cell_count=counts[g*N_CELLS:(g+1)*N_CELLS].min()))
    households = pd.factorize(people.SERIALNO, sort=False)[0]
    by_score = np.argsort(score,kind="stable")
    orders = [by_score, by_score[group[by_score] == 0], by_score[group[by_score] == 1]]
    return Cohort(score, group, cells, households, orders, summaries, counts)


def scenario_mass(current, previous):
    p = current.reshape(2, N_CELLS) / current.sum()
    r = previous.reshape(2, N_CELLS) / previous.sum()
    if np.any((p == 0) & (r > 0)):
        raise ValueError("Reference cell has no current support")
    p_group, r_group = p.sum(axis=1, keepdims=True), r.sum(axis=1, keepdims=True)
    return (p, r_group*p/p_group, p_group*r/r_group, r)


def single_margin_mass(current, previous):
    """Match one previous-year margin within each group; keep current group shares."""
    p = current.reshape(2, N_CELLS) / current.sum()
    r = previous.reshape(2, N_CELLS) / previous.sum()
    levels = np.arange(N_CELLS)
    factors = {"age_band":levels//8, "sex":(levels//4)%2,
               "education":(levels//2)%2, "weekly_hours":levels%2}
    masses = {}
    for name, factor in factors.items():
        mass = p.copy()
        for group in (0, 1):
            source = np.bincount(factor, weights=p[group]/p[group].sum())
            target = np.bincount(factor, weights=r[group]/r[group].sum())
            mass[group] *= (target/source)[factor]
            np.testing.assert_allclose(np.bincount(factor, weights=mass[group]/mass[group].sum()),
                                       target, atol=1e-12)
        masses[name] = mass
    return masses


def quantile_from_cumulative(sorted_scores, cumulative, n):
    """Linear sample quartile with the two rank scores found in the weighted CDF.

    At p=.25, k=floor((n-1)p). Interpolate the scores at CDF levels (k+1)/n
    and (k+2)/n with fractional part of (n-1)p. With equal weights this is
    numpy.quantile's linear rule. The rank resolution n is the count of sampled
    people in the respective pool or group before standardization.
    """
    position = (n-1)*.25
    lo = int(np.floor(position))
    levels = (np.array([lo+1, min(n, lo+2)]) / n) * cumulative[-1]
    # Repeated addition of a uniform fractional weight can place an exact rank
    # infinitesimally below its requested mass. This tolerance is sub-person.
    tol = 8*np.finfo(float).eps*n*cumulative[-1]
    idx = np.searchsorted(cumulative, levels-tol, side="left")
    scores = sorted_scores[np.minimum(idx, len(sorted_scores)-1)]
    return scores[0] + (position-lo)*(scores[1]-scores[0])


def evaluate(cohort, betas, frequency, prior_counts, sorted_scores, return_details=False,
             specified_masses=None):
    current = np.bincount(cohort.cells, weights=frequency, minlength=2*N_CELLS)
    current_mix = current / current.sum()
    sizes = current.reshape(2, N_CELLS).sum(axis=1).astype(int)
    sample_sizes = [int(current.sum()), *sizes]
    estimates, extras = [], []
    masses = scenario_mass(current, prior_counts) if specified_masses is None else specified_masses
    for mass in masses:
        ratio = np.divide(mass.ravel(), current_mix, out=np.zeros_like(current_mix), where=current_mix>0)
        weights = ratio[cohort.cells] * frequency
        cum = [np.cumsum(weights[o]) for o in cohort.orders]
        pooled, white, black = [quantile_from_cumulative(s, c, n) for s, c, n in
                                zip(sorted_scores, cum, sample_sizes)]
        cutoffs = (1-betas)*pooled + betas*np.array([[white], [black]])
        proportions = []
        for g in (0, 1):
            positions = np.searchsorted(sorted_scores[1+g], cutoffs[g], side="left")
            rejected = np.where(positions == 0, 0, cum[1+g][positions-1])
            proportions.append(1-rejected/cum[g+1][-1])
        estimates.append(100*(proportions[1]-proportions[0]))
        if return_details:
            empirical = np.bincount(cohort.cells, weights=weights, minlength=2*N_CELLS)/weights.sum()
            np.testing.assert_allclose(empirical, mass.ravel(), atol=1e-10, rtol=0)
            group_indices = [cohort.orders[g+1] for g in (0, 1)]
            ess = [weights[i].sum()**2 / np.square(weights[i]).sum() for i in group_indices]
            extras.append(dict(pooled_q25=pooled, white_q25=white, black_q25=black,
                               black_share=mass[1].sum(), max_weight=ratio.max(),
                               min_weight=ratio.min(), ess0=ess[0], ess1=ess[1],
                               tau0=cutoffs[0],tau1=cutoffs[1],
                               approval0_pp=100*proportions[0], approval1_pp=100*proportions[1]))
    return np.array(estimates), extras


def household_draw(codes, rng):
    n = int(codes.max())+1
    return np.bincount(rng.integers(n,size=n),minlength=n)[codes]


def state_results(root, state, draws):
    transfer = root / "results/transfer/from2010"
    policies = pd.concat([pd.read_csv(transfer / name) for name in
                          ("rolling_previous.csv","rolling_state_year.csv")])
    policies = policies[policies.state == state].set_index(["year","policy"])
    prior = read_cohort(root, state, YEARS[0])
    summary = prior.summaries.copy()
    contrasts, intervals, strata, margins = [], [], [], []

    def save_strata(year, counts):
        for c, n in enumerate(counts):
            group, cell = divmod(c, N_CELLS)
            strata.append(dict(state=state,year=year,group=("White","Black")[group],
                               age_band=("17-34","35-54","55+")[cell//8],female=(cell//4)%2,
                               bachelor=(cell//2)%2, weekly_hours_ge35=cell%2, n=int(n)))

    save_strata(YEARS[0], prior.cell_counts)
    for ref, year in zip(YEARS[:-1], YEARS[1:]):
        current = read_cohort(root,state,year)
        summary.extend(current.summaries)
        save_strata(year, current.cell_counts)
        frozen = policies.loc[year].loc[list(POLICIES)]
        betas = frozen.beta.to_numpy()
        ordered = [current.score[i] for i in current.orders]
        actual, extras = evaluate(current, betas, np.ones(current.n), prior.cell_counts,
                                  ordered, return_details=True)
        np.testing.assert_allclose(actual[0],frozen.gap_pp,atol=1e-10,rtol=0)
        for g in (0,1):
            tau = f"tau{g}"
            np.testing.assert_allclose(extras[0][tau], frozen[tau], atol=2e-14,rtol=0)
            np.testing.assert_allclose(extras[0][f"approval{g}_pp"],
                                       100*frozen[f"approved{g}"]/frozen[f"n{g}"],atol=1e-10,rtol=0)
        for j,name in enumerate(SCENARIOS):
            for i,policy in enumerate(POLICIES):
                info = {k:(v[i] if isinstance(v,np.ndarray) else v) for k,v in extras[j].items()}
                contrasts.append(dict(state=state,year=year,reference_year=ref,policy=policy,
                                      scenario=name,beta=betas[i],n=current.n,
                                      n0=int(frozen.n0.iloc[0]),n1=int(frozen.n1.iloc[0]),
                                      gap_pp=actual[j,i],delta_from_observed_pp=actual[j,i]-actual[0,i],
                                      min_current_cell=current.cell_counts.min(),
                                      min_reference_cell=prior.cell_counts.min(),**info))
        single = single_margin_mass(current.cell_counts, prior.cell_counts)
        marginal,_ = evaluate(current,betas,np.ones(current.n),prior.cell_counts,ordered,
                              specified_masses=single.values())
        for j,name in enumerate(single):
            for i,policy in enumerate(POLICIES):
                margins.append(dict(state=state,year=year,reference_year=ref,policy=policy,
                                    factor=name,observed_gap_pp=actual[0,i],
                                    standardized_gap_pp=marginal[j,i],
                                    delta_from_observed_pp=marginal[j,i]-actual[0,i]))
        if draws:
            rng = np.random.default_rng([1982,2021,year,STATES.index(state)])
            replicates = []
            for _ in range(draws):
                frequency = household_draw(current.households,rng)
                prior_counts = np.bincount(prior.cells, weights=household_draw(prior.households,rng),
                                           minlength=2*N_CELLS)
                rep,_ = evaluate(current,betas,frequency,prior_counts,ordered)
                replicates.append(rep)
            reps = np.stack(replicates)
            low,high = np.quantile(reps,[.025,.975],axis=0)
            dlo,dhi = np.quantile(reps-reps[:,:1,:],[.025,.975],axis=0)
            for j,name in enumerate(SCENARIOS):
                for i,policy in enumerate(POLICIES):
                    intervals.append(dict(state=state,year=year,reference_year=ref,policy=policy,
                                          scenario=name,draws=draws,lower_pp=low[j,i],upper_pp=high[j,i],
                                          delta_lower_pp=dlo[j,i],delta_upper_pp=dhi[j,i]))
        print(f"{state} {year} standardizations complete ({draws} resamples)",flush=True)
        prior = current
    return contrasts,summary,strata,intervals,margins


def main(root,draws,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(state_results,root,s,draws) for s in STATES]
        data = [job.result() for job in jobs]
    out = root / "results/diagnostics"
    out.mkdir(parents=True,exist_ok=True)
    for name, records in zip(("composition_contrasts","cohort_composition","composition_cells",
                              "composition_intervals","composition_single_margin"),zip(*data)):
        rows = [row for state in records for row in state]
        if rows:
            target = out/(name+".csv")
            temp = target.with_suffix(".partial")
            pd.DataFrame(rows).to_csv(temp,index=False)
            temp.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--draws",type=int,default=400,
                        help="Household bootstrap replicates per fold; zero writes point estimates only.")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.draws < 0 or args.workers < 1:
        parser.error("draws must be nonnegative and workers must be positive")
    main(args.root.resolve(),args.draws,args.workers)
