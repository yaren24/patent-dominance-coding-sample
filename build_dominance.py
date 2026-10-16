"""
Part A: how dominant is each inventor in her technology field?

Reads four PatentsView bulk tables and writes, for every inventor, field and year t, her share
of the field's patents filed in the five years t-4 to t, her rank in the field, and, for the
leader, the dominance margin (the leader's share divided by the runner-up's).

    python build_dominance.py --data-dir data/patentsview --out-dir output

Reads  (unzipped PatentsView .tsv files, kept as downloaded):
    g_patent.tsv, g_application.tsv, g_inventor_disambiguated.tsv, g_cpc_current.tsv
Writes:
    output/inventor_field_years.parquet  one row per inventor x field x year
    output/field_years.parquet           one row per field x year

The rules, each fixed in the settings block below:
    - a patent is dated by its application year, when the invention was made;
    - only utility patents, not withdrawn, filed 1976 to 2022;
    - a field is the CPC main group of the patent's primary code (e.g. A61K31/00);
    - fractional counting: a patent with n inventors gives 1/n to each, so shares sum to one
      and nobody becomes dominant just by sitting on large teams.
"""

import argparse
import time
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Settings: every analytical choice in this file
# ---------------------------------------------------------------------------

FIRST_FILING_YEAR = 1976   # PatentsView holds every patent granted from 1976, so 1976 is the
                           # first filing year whose field totals are complete
LAST_FILING_YEAR = 2022    # later filings are still in examination, so their years look empty
PATENT_TYPES = ("utility",)
PRIMARY_CPC_SEQUENCE = 0   # g_cpc_current numbers a patent's codes from 0; 0 is the primary one
WINDOW_YEARS = 5           # a share is measured over the filing years t-4 to t
FIRST_WINDOW_YEAR = FIRST_FILING_YEAR + WINDOW_YEARS - 1   # 1980: the first complete window
LAST_WINDOW_YEAR = LAST_FILING_YEAR
WEIGHT_DECIMALS = 9        # window weights are rounded so that equal sums tie exactly

RAW_FILES = {
    "patents": "g_patent.tsv",
    "applications": "g_application.tsv",
    "inventors": "g_inventor_disambiguated.tsv",
    "cpc": "g_cpc_current.tsv",
}


# ---------------------------------------------------------------------------
# 1. Reading the raw tables
# ---------------------------------------------------------------------------

def read_tsv(path):
    """
    A PatentsView .tsv file as DuckDB reads it. Every column stays text, so that nothing is
    converted (or silently set to missing) before the code below decides how to convert it.
    """
    return f"""read_csv('{Path(path).as_posix()}', delim = '\t', header = true, quote = '"',
                        escape = '"', all_varchar = true, sample_size = -1, null_padding = true)"""


def count(con, table):
    return con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]


def keep_rows(con, table, condition, reason):
    """
    Keep the rows of `table` that meet `condition` (SQL), and print how many were dropped and
    why. No row leaves the sample without a line in the log saying so.
    """
    before = count(con, table)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM {table} WHERE {condition}")
    print(f"  {before - count(con, table):>12,} dropped: {reason}")


# ---------------------------------------------------------------------------
# 2. Patents, fields and the inventor spine
# ---------------------------------------------------------------------------

def build_patent_fields(con, data_dir):
    """One row per patent: its field, the CPC main group of its primary code."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cpc AS
        SELECT patent_id, cpc_sequence AS cpc_sequence_text, cpc_group
        FROM {read_tsv(Path(data_dir) / RAW_FILES["cpc"])}
    """)
    # A sequence that is not a whole number is a data problem to look at, not to cast away.
    bad = con.sql("""SELECT count(*) FROM cpc WHERE cpc_sequence_text IS NOT NULL
                     AND TRY_CAST(cpc_sequence_text AS INTEGER) IS NULL""").fetchone()[0]
    if bad:
        raise ValueError(f"{bad:,} CPC rows have a sequence that is not a whole number.")

    # PatentsView stores the full subgroup (A61K31/4402). The main group is the part before
    # the slash with /00 restored: A61K31/00.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE patent_fields AS
        SELECT patent_id, split_part(cpc_group, '/', 1) || '/00' AS field_id
        FROM cpc
        WHERE CAST(cpc_sequence_text AS INTEGER) = {PRIMARY_CPC_SEQUENCE}
    """)
    if count(con, "patent_fields") == 0:
        raise ValueError(f"No CPC row has sequence {PRIMARY_CPC_SEQUENCE}: check which CPC "
                         f"table was read, because each numbers its codes differently.")
    keep_rows(con, "patent_fields", "field_id IS NOT NULL AND field_id <> ''",
              "no usable CPC main group, so the patent cannot be placed in a field")


def build_patents(con, data_dir):
    """One row per patent in the universe, dated by its application year."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE patents AS
        SELECT p.patent_id,
               p.patent_type,
               COALESCE(TRY_CAST(p.withdrawn AS INTEGER), 0) AS withdrawn,
               year(TRY_CAST(a.filing_date AS DATE))       AS year
        FROM {read_tsv(Path(data_dir) / RAW_FILES["patents"])} p
        LEFT JOIN {read_tsv(Path(data_dir) / RAW_FILES["applications"])} a
               ON a.patent_id = p.patent_id
    """)
    types = ", ".join(f"'{t}'" for t in PATENT_TYPES)
    keep_rows(con, "patents", f"patent_type IN ({types})",
              "design, plant and reissue patents are not new inventions")
    keep_rows(con, "patents", "withdrawn = 0", "withdrawn by the patent office")
    keep_rows(con, "patents", "year IS NOT NULL", "no usable application date")
    keep_rows(con, "patents", f"year >= {FIRST_FILING_YEAR}",
              f"filed before {FIRST_FILING_YEAR}: only the fast-granted ones are in the data")
    keep_rows(con, "patents", f"year <= {LAST_FILING_YEAR}",
              f"filed after {LAST_FILING_YEAR}: many are still in examination")
    keep_rows(con, "patents", "patent_id IN (SELECT patent_id FROM patent_fields)",
              "no field, so the patent counts towards no field's total")


def build_inventor_patents(con, data_dir):
    """
    One row per inventor listing on a kept patent, with her fractional credit 1/team size.
    The team is counted after the patent filter, which removes whole patents only, so every
    team is intact.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE inventor_patents AS
        WITH listings AS (
            SELECT patent_id, CAST(inventor_sequence AS INTEGER) AS inventor_sequence,
                   inventor_id, count(*) OVER (PARTITION BY patent_id) AS team_size
            FROM {read_tsv(Path(data_dir) / RAW_FILES["inventors"])}
            WHERE patent_id IN (SELECT patent_id FROM patents)
        )
        SELECT l.patent_id, l.inventor_id, l.inventor_sequence, l.team_size,
               f.field_id, p.year, 1.0 / l.team_size AS weight
        FROM listings l
        JOIN patents p       ON p.patent_id = l.patent_id
        JOIN patent_fields f ON f.patent_id = l.patent_id
    """)

    # Every patent's credits must add to exactly one. If not, every share below is wrong in
    # a way that still looks plausible.
    off = con.sql("""SELECT count(*) FROM (SELECT patent_id FROM inventor_patents GROUP BY 1
                     HAVING abs(sum(weight) - 1.0) > 1e-9)""").fetchone()[0]
    if off:
        raise ValueError(f"{off:,} patents have credits that do not sum to 1.")

    no_inventor = con.sql("""SELECT count(*) FROM patents
                             WHERE patent_id NOT IN (SELECT patent_id FROM inventor_patents)""")
    print(f"  {no_inventor.fetchone()[0]:>12,} patents have no inventor record: they still count "
          f"in their field's patent total, but give no one any credit")


# ---------------------------------------------------------------------------
# 3. The backward-looking window
# ---------------------------------------------------------------------------

def build_windows(con):
    """
    Each inventor's credit in each field over the filing years t-4 to t.

    The window is built forwards: a patent filed in year f is sent to the windows f, f+1, ...,
    f+4. So a patent can never reach a window that closed before it was filed, which is the
    look-ahead an event study must rule out.
    """
    # A person listed twice on one patent is one inventor holding two shares of it, and one
    # patent in her patent count.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE activity AS
        WITH credit AS (
            SELECT patent_id, inventor_id, field_id, year AS filing_year,
                   count(*) * (1.0 / team_size) AS weight
            FROM inventor_patents
            GROUP BY patent_id, inventor_id, team_size, field_id, year
        )
        SELECT inventor_id, field_id, filing_year,
               sum(weight) AS weight_in_year,
               count(*)    AS patents_in_year
        FROM credit
        GROUP BY 1, 2, 3
    """)
    window_rows = f"""
        SELECT a.*, a.filing_year + step.years_ahead AS year
        FROM activity a, (SELECT unnest(range(0, {WINDOW_YEARS})) AS years_ahead) step
    """
    future = con.sql(f"SELECT count(*) FROM ({window_rows}) WHERE filing_year > year")
    if future.fetchone()[0]:
        raise ValueError("A patent entered a window that closed before it was filed.")

    # Rounding matters: 1/2 + 1/3 + 1/6 and 1/2 + 1/2 are both one patent, but not equal in
    # floating point. Unrounded, a real tie at the top would become a strict ranking.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE windows AS
        SELECT inventor_id, field_id, year,
               year - {WINDOW_YEARS} + 1                          AS window_first_year,
               round(sum(weight_in_year), {WEIGHT_DECIMALS})      AS inventor_weight_in_window,
               sum(patents_in_year)                               AS inventor_patents_in_window
        FROM ({window_rows})
        GROUP BY 1, 2, 3
    """)
    keep_rows(con, "windows", f"year BETWEEN {FIRST_WINDOW_YEAR} AND {LAST_WINDOW_YEAR}",
              f"windows ending before {FIRST_WINDOW_YEAR} hold fewer than {WINDOW_YEARS} "
              f"years of data")


# ---------------------------------------------------------------------------
# 4. Share, rank and margin
# ---------------------------------------------------------------------------

RANKED_SQL = """
    -- The denominator is a window sum over the numerator column itself, so the share's
    -- numerator and denominator cannot use different counting rules.
    -- rank(): tied inventors share a rank, and the next rank is skipped (1, 1, 3).
    -- position: a unique sort order, used only to find the runner-up's weight.
    SELECT *,
           sum(inventor_weight_in_window) OVER (PARTITION BY field_id, year)
               AS field_weight_in_window,
           inventor_weight_in_window
               / sum(inventor_weight_in_window) OVER (PARTITION BY field_id, year) AS share,
           rank() OVER (PARTITION BY field_id, year ORDER BY inventor_weight_in_window DESC)
               AS rank,
           count(*) OVER (PARTITION BY field_id, year, inventor_weight_in_window)
               AS n_tied_at_rank,
           row_number() OVER (PARTITION BY field_id, year
                              ORDER BY inventor_weight_in_window DESC, inventor_id) AS position
    FROM windows
"""

MARGINS_SQL = """
    -- The margin belongs to the field-year: the top weight over the second row's weight.
    -- A tied top therefore has a margin of exactly 1: if two people are level, neither towers.
    -- A field-year with one inventor has no runner-up, so its margin is missing, not invented.
    SELECT field_id, year,
           top_weight / second_weight     AS margin_uncapped,
           second_weight IS NOT NULL      AS has_runner_up
    FROM (
        SELECT field_id, year,
               max(inventor_weight_in_window) FILTER (WHERE position = 1) AS top_weight,
               max(inventor_weight_in_window) FILTER (WHERE position = 2) AS second_weight
        FROM ranked
        GROUP BY 1, 2
    )
"""


def build_inventor_field_years(con):
    con.execute(f"CREATE OR REPLACE TEMP TABLE ranked AS {RANKED_SQL}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE margins AS {MARGINS_SQL}")
    # The margin is stored on rank-1 rows only, and uncapped: a cap is a choice for later.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE inventor_field_years AS
        SELECT r.inventor_id, r.field_id, r.year, r.window_first_year,
               r.inventor_weight_in_window, r.inventor_patents_in_window,
               r.field_weight_in_window, r.share, r.rank, r.n_tied_at_rank,
               r.rank = 1                                       AS is_top,
               CASE WHEN r.rank = 1 THEN m.margin_uncapped END  AS margin_uncapped,
               CASE WHEN r.rank = 1 THEN m.has_runner_up END    AS has_runner_up
        FROM ranked r
        JOIN margins m ON m.field_id = r.field_id AND m.year = r.year
    """)


def build_field_years(con):
    """
    One row per field-year, read back from the inventor table so that the two cannot disagree.
    The patent head count is a separate quantity: it includes patents with no inventor record,
    and it is never a share's denominator.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE field_patent_counts AS
        SELECT f.field_id, p.year + step.years_ahead AS year, count(*) AS field_patents_in_window
        FROM patent_fields f
        JOIN patents p ON p.patent_id = f.patent_id,
             (SELECT unnest(range(0, {WINDOW_YEARS})) AS years_ahead) step
        WHERE p.year + step.years_ahead BETWEEN {FIRST_WINDOW_YEAR} AND {LAST_WINDOW_YEAR}
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE field_years AS
        SELECT i.field_id, i.year,
               any_value(i.window_first_year)                   AS window_first_year,
               any_value(i.field_weight_in_window)              AS field_weight_in_window,
               any_value(c.field_patents_in_window)             AS field_patents_in_window,
               count(*)                                         AS inventors_in_field_window,
               max(i.share)                                     AS top_share,
               max(i.n_tied_at_rank) FILTER (WHERE i.rank = 1)  AS n_tied_at_top,
               max(i.margin_uncapped)                           AS margin_uncapped,
               bool_or(i.has_runner_up)                         AS has_runner_up
        FROM inventor_field_years i
        LEFT JOIN field_patent_counts c ON c.field_id = i.field_id AND c.year = i.year
        GROUP BY 1, 2
    """)


# ---------------------------------------------------------------------------
# 5. Checks: any failure stops the run before anything is written
# ---------------------------------------------------------------------------

CHECKS = {
    # A wrong denominator, double counting, or mixed counting rules all break this one.
    "shares sum to 1 in every field-year": """
        SELECT coalesce(max(abs(total - 1.0)) < 1e-9, false)
        FROM (SELECT sum(share) AS total FROM inventor_field_years GROUP BY field_id, year)""",
    "every share lies in [0, 1]": """
        SELECT count(*) = 0 FROM inventor_field_years WHERE share < 0 OR share > 1""",
    "every field-year has a rank-1 inventor": """
        SELECT count(*) = 0 FROM (SELECT field_id, year FROM inventor_field_years
                                  GROUP BY 1, 2 HAVING count(*) FILTER (WHERE rank = 1) = 0)""",
    "every margin is at least 1": """
        SELECT count(*) = 0 FROM inventor_field_years WHERE margin_uncapped < 1""",
    "one row per inventor x field x year": """
        SELECT count(*) = 0 FROM (SELECT inventor_id, field_id, year FROM inventor_field_years
                                  GROUP BY 1, 2, 3 HAVING count(*) > 1)""",
    # The windows only re-arrange credit: the total handed out must not change.
    "total credit equals the inventor table's": """
        SELECT abs((SELECT sum(weight_in_year) FROM activity)
                   - (SELECT sum(weight) FROM inventor_patents)) < 1e-3""",
}


def run_checks(con):
    failed = [name for name, sql in CHECKS.items() if not con.sql(sql).fetchone()[0]]
    for name in CHECKS:
        print(f"  {'FAIL' if name in failed else 'pass'}  {name}")
    if failed:
        raise ValueError(f"{len(failed)} check(s) failed: {failed}. Nothing was written.")


# ---------------------------------------------------------------------------
# 6. The run
# ---------------------------------------------------------------------------

def run(data_dir, out_dir, memory_limit="2GB"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{(out_dir / 'duckdb_tmp').as_posix()}'")

    print("Patents and fields")
    build_patent_fields(con, data_dir)
    build_patents(con, data_dir)
    build_inventor_patents(con, data_dir)
    print("Windows")
    build_windows(con)
    build_inventor_field_years(con)
    build_field_years(con)
    print("Checks")
    run_checks(con)

    for table, key in [("inventor_field_years", "field_id, year, inventor_id"),
                       ("field_years", "field_id, year")]:
        path = (out_dir / f"{table}.parquet").as_posix()
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY {key}) TO '{path}' (FORMAT parquet)")
        print(f"  wrote {count(con, table):,} rows to {path}")
    return con


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder with the four .tsv files")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--memory-limit", default="2GB", help="DuckDB spills to disk above this")
    args = parser.parse_args()
    started = time.perf_counter()
    con = run(args.data_dir, args.out_dir, args.memory_limit)
    top = con.sql("""SELECT round(quantile_cont(top_share, 0.5), 4),
                            round(quantile_cont(top_share, 0.9), 4)
                     FROM field_years""").fetchone()
    print(f"Median leader's share {top[0]}, 90th percentile {top[1]}. "
          f"Run time {time.perf_counter() - started:.0f} s.")


if __name__ == "__main__":
    main()
