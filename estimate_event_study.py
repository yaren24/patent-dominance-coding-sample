"""
Part B: did the field's patenting change after its leading inventor died?

A stacked event study. Each "stack" is one deceased leader's field (treated) together with
its matched control fields, observed from five years before her death (D-5) to five years
after (D+5). A weighted Poisson regression compares the treated field's yearly patent count
with its controls', year by year relative to D-1, with:

    - a fixed effect for each field within its stack (its own level of patenting);
    - a fixed effect for each stack and year (anything common to the stack in that year);
    - standard errors clustered two ways, by the deceased and by the field, because one person
      can lead several fields and one field can serve as a control in several stacks.

    PYTHONHASHSEED=0 python estimate_event_study.py --panel outcome_panel.parquet

The input panel is built by earlier steps of the thesis (deaths, matching, outcomes), which
are not part of this sample. It has one row per stack, field and year, with these columns:

    inventor_id             the deceased leader (PatentsView id)
    death_year              her death year D
    field_id                her field: with inventor_id and death_year, it names the stack
    unit_field_id           the field this row measures: her field, or one of its controls
    role                    'treated' or 'control'
    years_from_death        -5 to +5
    outcome                 patents filed in unit_field_id that year
    pair_weight             1 on the treated field; 1 / (number of controls) on a control
    deceased_matched_pairs  how many of her fields were matched (stacks of hers)

Effects are reported as exp(b) - 1, the percent change in patenting.
"""

import argparse
import os
import re
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

YEARS_BEFORE_DEATH = 5
YEARS_AFTER_DEATH = 5
REFERENCE_YEAR = -1             # every coefficient is measured against D-1
SIGNIFICANCE_LEVEL = 0.05
YEARS = list(range(-YEARS_BEFORE_DEATH, YEARS_AFTER_DEATH + 1))

PANEL_COLUMNS = ["inventor_id", "death_year", "field_id", "unit_field_id", "role",
                 "years_from_death", "outcome", "pair_weight", "deceased_matched_pairs"]


# ---------------------------------------------------------------------------
# 1. The terms of the two specifications
# ---------------------------------------------------------------------------

def term_name(k):
    """treated_year_m5 for D-5, treated_year_0 for D, treated_year_p3 for D+3."""
    if k < 0:
        return f"treated_year_m{-k}"
    return "treated_year_0" if k == 0 else f"treated_year_p{k}"


def event_terms():
    """The event study: one treated x year term for every year except the reference, D-1."""
    return [(term_name(k), k, k) for k in YEARS if k != REFERENCE_YEAR]


def pooled_terms():
    """The pooled fit: D+1 to D+5 together, and D apart. The reference is D-5 to D-1."""
    return [("treated_after", 1, YEARS_AFTER_DEATH), ("treated_death_year", 0, 0)]


def pre_trend_terms():
    """D-5 to D-2: if the design is sound, none of these differs from D-1."""
    return [name for name, first, _ in event_terms() if first < 0]


def event_periods():
    """The periods the separation guard checks for the event study: every year, D-1 included."""
    return [(k, k) for k in YEARS]


def pooled_periods():
    """The same for the pooled fit: its reference D-5 to D-1, then D, then D+1 to D+5."""
    return [(YEARS[0], REFERENCE_YEAR), (0, 0), (1, YEARS_AFTER_DEATH)]


# ---------------------------------------------------------------------------
# 2. The panel and its checks
# ---------------------------------------------------------------------------

SHAPE_SQL = """
    -- Each field-stack has one row in every year D-5 to D+5 and one role; each stack has
    -- exactly one treated field.
    WITH field_stacks AS (
        SELECT death_year, inventor_id, field_id, unit_field_id,
               count(*) AS rows, count(DISTINCT years_from_death) AS years,
               min(years_from_death) AS first_year, max(years_from_death) AS last_year,
               count(DISTINCT role) AS roles, bool_or(role = 'treated') AS treated
        FROM panel GROUP BY ALL
    ), stacks AS (
        SELECT death_year, inventor_id, field_id,
               count(*) FILTER (WHERE treated) AS treated_field_stacks
        FROM field_stacks GROUP BY ALL
    )
    SELECT
        (SELECT count(*) FROM field_stacks
         WHERE rows <> $years OR years <> $years OR first_year <> $first OR last_year <> $last),
        (SELECT count(*) FROM field_stacks WHERE roles <> 1),
        (SELECT count(*) FROM stacks WHERE treated_field_stacks <> 1),
        (SELECT count(*) FROM panel WHERE role NOT IN ('treated', 'control'))
"""


def read_panel(con, path):
    """The panel as a DuckDB view, restricted to the columns used, with its shape checked."""
    reader = "read_parquet" if str(path).endswith(".parquet") else "read_csv"
    quoted = str(path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT {', '.join(PANEL_COLUMNS)} "
                f"FROM {reader}('{quoted}')")
    # pyfixest would drop a row with a missing value silently. Stop instead.
    result = con.sql("SELECT count(*) - count(COLUMNS(*)) FROM panel")
    missing = {c: n for c, n in zip(result.columns, result.fetchone()) if n}
    if missing:
        raise ValueError(f"Missing values in the panel: {missing}.")
    problems = con.execute(SHAPE_SQL, {"years": len(YEARS), "first": YEARS[0],
                                       "last": YEARS[-1]}).fetchone()
    labels = ["field-stacks without one row per year", "field-stacks with two roles",
              "stacks without exactly one treated field", "rows with an unknown role"]
    found = {label: n for label, n in zip(labels, problems) if n}
    if found:
        raise ValueError(f"The panel is not a stacked panel: {found}.")


FRAME_SQL = """
    -- The weight makes each death count once per side in every year: her treated field
    -- weighs 1 / (her stacks); her controls share the same total.
    -- Integer codes for the fixed effects and clusters: a stack is (death_year, inventor_id,
    -- field_id); a field-stack adds the unit's own field; a stack-year adds the year.
    SELECT *,
           pair_weight / deceased_matched_pairs                                 AS weight,
           CAST(role = 'treated' AS INTEGER)                                    AS treated,
           dense_rank() OVER (ORDER BY death_year, inventor_id, field_id)       AS stack_code,
           dense_rank() OVER (ORDER BY death_year, inventor_id, field_id, unit_field_id)
                                                                                AS field_stack_code,
           dense_rank() OVER (ORDER BY death_year, inventor_id, field_id, years_from_death)
                                                                                AS stack_year_code,
           dense_rank() OVER (ORDER BY inventor_id)                             AS deceased_code,
           dense_rank() OVER (ORDER BY unit_field_id)                           AS field_code
    FROM panel
    ORDER BY death_year, inventor_id, field_id, role DESC, unit_field_id, years_from_death
"""


def regression_frame(con):
    """The panel with the weight, the integer codes, and both specifications' terms."""
    frame = con.sql(FRAME_SQL).df()
    k = frame["years_from_death"]
    for name, first, last in event_terms() + pooled_terms():
        frame[name] = frame["treated"] * k.between(first, last).astype(int)
    return frame


def check_weights(frame):
    """
    Each death must count once per side in every year: weights positive, 1 on treated pairs,
    the treated weight equal to its controls' total in every stack-year, and each side summing
    to the number of deaths in every year.
    """
    def close(a, b):
        return np.isclose(a, b, rtol=1e-9, atol=0.0).all()

    deaths = frame["inventor_id"].nunique()
    side = frame.groupby(["years_from_death", "role"])["weight"].sum()
    stack_year = frame.pivot_table(index="stack_year_code", columns="role", values="weight",
                                   aggfunc="sum", fill_value=0.0)
    problems = []
    if not (frame["weight"] > 0).all():
        problems.append("a weight is not above 0")
    if not close(frame.loc[frame["treated"] == 1, "pair_weight"], 1.0):
        problems.append("pair_weight is not 1 on a treated row")
    if not close(stack_year["treated"], stack_year["control"]):
        problems.append("a stack-year's treated weight differs from its controls' total")
    if not close(side, deaths):
        problems.append(f"a side's weights do not sum to the {deaths} deaths in some year")
    if problems:
        raise ValueError("The weights are wrong: " + "; ".join(problems) + ".")


# ---------------------------------------------------------------------------
# 3. Rows that carry no information
# ---------------------------------------------------------------------------

def remove_rows_without_information(frame):
    """
    Remove, pass after pass until nothing changes:
      (i)   field-stacks with zero patents in every year,
      (ii)  stack-years with zero patents in every row,
      (iii) field-stacks or stack-years left with a single row.
    Each is fitted exactly by its own fixed effect, so it tells the regression nothing about the
    treated x year terms; dropping it leaves the estimate unchanged. pyfixest would drop some of
    these itself, without saying which, so we do it first and later require it to drop nothing.
    Removing one kind can create another, hence the loop.
    """
    removed, pass_number = [], 0
    while True:
        pass_number += 1
        removed_this_pass = 0
        for kind in ("all_zero_field_stacks", "all_zero_stack_years", "singletons"):
            by_field_stack = frame.groupby("field_stack_code")["outcome"]
            by_stack_year = frame.groupby("stack_year_code")["outcome"]
            if kind == "all_zero_field_stacks":
                keep = by_field_stack.transform("sum") > 0
            elif kind == "all_zero_stack_years":
                keep = by_stack_year.transform("sum") > 0
            else:
                keep = (by_field_stack.transform("size") > 1) & (by_stack_year.transform("size") > 1)
            removed.append({"pass": pass_number, "kind": kind, "rows": int((~keep).sum())})
            removed_this_pass += int((~keep).sum())
            frame = frame[keep]
        if removed_this_pass == 0:
            kept = frame.sort_values(["stack_code", "field_stack_code", "years_from_death"])
            return kept.reset_index(drop=True), removed


# ---------------------------------------------------------------------------
# 4. The separation guard and the fit
# ---------------------------------------------------------------------------

def check_separation(frame, periods):
    """
    A term whose treated rows all have zero patents is "separated": its coefficient runs to
    minus infinity, yet pyfixest stops near -20 and reports convergence. So every period, the
    reference included (with none there, every other term runs to plus infinity), must keep at
    least one treated row with a positive count.
    """
    positive_treated = (frame["treated"] == 1) & (frame["outcome"] > 0)
    empty = [(first, last) for first, last in periods
             if not (positive_treated & frame["years_from_death"].between(first, last)).any()]
    if empty:
        raise ValueError(f"Separation: no treated row has a positive outcome in years {empty} "
                         f"relative to the death.")


def fit_poisson(frame, terms, periods):
    """
    The weighted Poisson with field-within-stack and stack x year fixed effects, and two-way
    clustered standard errors (CRV1, by deceased and by field). Every failure pyfixest reports
    only in a flag is turned into a stop, so a bad fit can never print a plausible table.
    """
    regressors = [name for name, _, _ in terms]
    check_separation(frame, periods)
    formula = f"outcome ~ {' + '.join(regressors)} | field_stack_code + stack_year_code"
    with warnings.catch_warnings():
        # pyfixest's own separation check only warns when it gives up; make that a stop.
        warnings.filterwarnings("error", message=re.escape("iterative rectivier separation"),
                                category=UserWarning)
        fit = pf.fepois(formula, data=frame, weights="weight",
                        vcov={"CRV1": "deceased_code+field_code"},
                        separation_check=["fe", "ir"])
    problems = []
    if not fit.convergence:
        problems.append("the fit did not converge")
    if fit._collin_vars:
        problems.append(f"{list(fit._collin_vars)} were dropped as collinear")
    if fit._N != len(frame):
        problems.append(f"pyfixest dropped {len(frame) - fit._N} rows we had not")
    # Both fixed effects are nested in the deceased cluster, so the small-sample correction
    # must count only the terms plus one as parameters. If pyfixest missed the nesting, every
    # standard error would be too large with nothing else looking wrong.
    if fit._df_k != len(regressors) + 1:
        problems.append(f"pyfixest counted K = {fit._df_k}, not {len(regressors) + 1}")
    vcov = pd.DataFrame(fit._vcov, index=fit._coefnames, columns=fit._coefnames)
    if not (np.diag(vcov) > 0).all():
        problems.append("a variance is not positive")
    if problems:
        raise ValueError("The fit failed: " + "; ".join(problems) + ".")
    return {"coefficients": pd.Series(fit.coef().to_numpy(), index=fit._coefnames),
            "vcov": vcov, "rows": fit._N, "deaths": frame["inventor_id"].nunique(),
            "fields": frame["unit_field_id"].nunique(), "stacks": frame["stack_code"].nunique()}


# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------

def normal_inference(result):
    """z, the two-sided p-value, the 95% interval, and the effect exp(b) - 1 with its interval."""
    beta = result["coefficients"]
    se = pd.Series(np.sqrt(np.diag(result["vcov"])), index=beta.index)
    z_critical = stats.norm.ppf(1 - SIGNIFICANCE_LEVEL / 2)
    lower, upper = beta - z_critical * se, beta + z_critical * se
    return pd.DataFrame({"beta": beta, "se": se, "p": 2 * stats.norm.sf((beta / se).abs()),
                         "effect": np.expm1(beta), "effect_lower": np.expm1(lower),
                         "effect_upper": np.expm1(upper)})


def wald_test(beta, vcov):
    """W = b' V^-1 b, chi-squared with as many degrees of freedom as coefficients."""
    beta, vcov = np.asarray(beta, dtype=float), np.asarray(vcov, dtype=float)
    if np.linalg.eigvalsh(vcov).min() <= 0:
        raise ValueError("The tested block of the variance matrix is not positive definite.")
    statistic = float(beta @ np.linalg.solve(vcov, beta))
    return statistic, len(beta), float(stats.chi2.sf(statistic, len(beta)))


# ---------------------------------------------------------------------------
# 6. The run
# ---------------------------------------------------------------------------

def check_hash_seed():
    """pyfixest's last digits depend on Python's string-hash seed; fix it so reruns match."""
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization:
        raise SystemExit("Run with PYTHONHASHSEED=0 so that a rerun gives identical digits.")


def estimate(panel_path):
    con = duckdb.connect()
    read_panel(con, panel_path)
    frame = regression_frame(con)
    check_weights(frame)
    kept, removed = remove_rows_without_information(frame)
    print(f"Rows: {len(frame):,} in, {sum(r['rows'] for r in removed):,} removed without "
          f"information, {len(kept):,} fitted.")
    event = fit_poisson(kept, event_terms(), event_periods())
    pooled = fit_poisson(kept, pooled_terms(), pooled_periods())
    pre = pre_trend_terms()
    w, df, p = wald_test(event["coefficients"][pre], event["vcov"].loc[pre, pre])
    return event, pooled, (w, df, p)


def print_results(event, pooled, pre_trend):
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\nDeaths {pooled['deaths']:,}; fields {pooled['fields']:,}; "
          f"stacks {pooled['stacks']:,}; rows {pooled['rows']:,}.")
    print("\nPooled: after D (D+1 to D+5) and D itself, against D-5 to D-1")
    print(normal_inference(pooled).to_string())
    print("\nEvent study, each year against D-1")
    print(normal_inference(event).to_string())
    w, df, p = pre_trend
    print(f"\nPre-death test, D-5 to D-2 jointly zero: W = {w:.3f}, {df} df, p = {p:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True, help="the outcome panel, .parquet or .csv")
    args = parser.parse_args()
    check_hash_seed()
    print_results(*estimate(args.panel))


if __name__ == "__main__":
    main()
