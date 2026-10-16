"""
Tests for estimate_event_study.py, on made-up stacked panels with a known effect.

What they protect against:
- an estimator that does not recover a known effect, or finds a pre-trend that is not there;
- pyfixest computing something other than the model described: its coefficients are checked
  against statsmodels with every fixed effect written out as a dummy;
- a separated term printing a plausible coefficient instead of stopping;
- a pre-death test that is not W = b' V^-1 b.
"""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import estimate_event_study as ees

TRUE_EFFECT = math.log(0.8)        # a 20% fall in patenting from D+1 on, nothing before
RECOVERY_TOLERANCE = 0.05          # about three standard errors on the big panel
COEFFICIENT_TOLERANCE = 1e-9       # pyfixest against statsmodels on the same rows


def made_up_panel(seed, n_deceased, mean_count, effect=TRUE_EFFECT):
    """
    A panel shaped like the real one: every fifth deceased leads two fields (two stacks), each
    stack has one to five controls drawn from a shared pool, so fields recur across stacks.
    """
    rng = np.random.default_rng(seed)
    rows, pool = [], max(4, 2 * n_deceased)
    for deceased in range(n_deceased):
        stacks = 2 if deceased % 5 == 0 else 1
        for _ in range(stacks):
            controls = int(rng.integers(1, 6))
            year_shock = rng.normal(0.0, 0.3, len(ees.YEARS))
            fields = rng.choice(pool, controls + 1, replace=False)
            for position, field in enumerate(fields):
                treated = position == 0
                level = math.log(mean_count) + rng.normal(0.0, 0.5)
                for j, k in enumerate(ees.YEARS):
                    log_mu = level + year_shock[j] + (effect if treated and k >= 1 else 0.0)
                    rows.append({"inventor_id": f"I{deceased}", "death_year": 1990 + deceased % 20,
                                 "field_id": f"F{fields[0]}", "unit_field_id": f"F{field}",
                                 "role": "treated" if treated else "control",
                                 "years_from_death": k,
                                 "outcome": int(rng.poisson(math.exp(log_mu))),
                                 "pair_weight": 1.0 if treated else 1.0 / controls,
                                 "deceased_matched_pairs": stacks})
    return pd.DataFrame(rows)


def write_panel(frame, path):
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT parquet)")
    return path


def fitted_rows(frame):
    con = duckdb.connect()
    con.register("panel", frame)
    kept, _ = ees.remove_rows_without_information(ees.regression_frame(con))
    return kept


@pytest.fixture(scope="module")
def big_panel(tmp_path_factory):
    path = tmp_path_factory.mktemp("panel") / "big.parquet"
    return write_panel(made_up_panel(seed=20261008, n_deceased=300, mean_count=40.0), path)


def test_the_pooled_fit_recovers_a_known_fall_after_the_death(big_panel):
    _, pooled, _ = ees.estimate(big_panel)
    assert pooled["coefficients"]["treated_after"] == pytest.approx(TRUE_EFFECT,
                                                                    abs=RECOVERY_TOLERANCE)
    assert pooled["coefficients"]["treated_death_year"] == pytest.approx(0.0,
                                                                         abs=RECOVERY_TOLERANCE)


def test_the_event_study_finds_no_pre_trend_and_the_fall_after_the_death(big_panel):
    event, _, (_, df, p) = ees.estimate(big_panel)
    for name, k, _ in ees.event_terms():
        truth = TRUE_EFFECT if k >= 1 else 0.0
        assert event["coefficients"][name] == pytest.approx(truth, abs=RECOVERY_TOLERANCE), k
    assert df == 4 and p > 0.01


def test_the_coefficients_equal_statsmodels_with_explicit_dummies():
    """The same Poisson, written out with one dummy per field-stack and per stack-year."""
    kept = fitted_rows(made_up_panel(seed=7, n_deceased=20, mean_count=12.0))
    ours = ees.fit_poisson(kept, ees.event_terms(), ees.event_periods())
    regressors = [name for name, _, _ in ees.event_terms()]
    field_stacks = pd.get_dummies(kept["field_stack_code"], prefix="fs", dtype=float)
    # Within a stack both sets of dummies sum to the stack's indicator, so one stack-year per
    # stack is left out to keep the design full rank.
    first_year = kept["years_from_death"] == kept["years_from_death"].min()
    stack_years = pd.get_dummies(kept["stack_year_code"].where(~first_year), prefix="sy",
                                 dtype=float)
    design = pd.concat([kept[regressors].astype(float), field_stacks, stack_years], axis=1)
    theirs = sm.GLM(kept["outcome"], design, family=sm.families.Poisson(),
                    var_weights=kept["weight"]).fit(tol=1e-13, maxiter=200)
    for name in regressors:
        assert ours["coefficients"][name] == pytest.approx(theirs.params[name],
                                                           rel=COEFFICIENT_TOLERANCE), name


def test_a_year_with_no_patent_on_the_treated_side_stops_the_fit():
    """Without the guard, pyfixest would return a coefficient near -20 and say it converged."""
    frame = made_up_panel(seed=7, n_deceased=20, mean_count=12.0)
    frame.loc[(frame["role"] == "treated") & (frame["years_from_death"] == 2), "outcome"] = 0
    kept = fitted_rows(frame)
    with pytest.raises(ValueError, match="Separation"):
        ees.fit_poisson(kept, ees.event_terms(), ees.event_periods())


def test_the_wald_test_equals_a_hand_computation():
    beta = np.array([0.10, -0.05])
    vcov = np.array([[0.004, 0.001], [0.001, 0.003]])
    statistic, df, _ = ees.wald_test(beta, vcov)
    by_hand = beta @ np.linalg.inv(vcov) @ beta
    assert df == 2 and statistic == pytest.approx(by_hand, rel=1e-12)
