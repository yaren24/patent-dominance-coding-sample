"""
Tests for build_dominance.py, on a tiny made-up PatentsView extract whose answers can be
worked out by hand.

What they protect against, in economic terms:
- shares that are not shares (a wrong denominator, double counting);
- look-ahead: a patent counted in a window that closed before it was filed;
- a tie at the top read as a towering leader;
- patents outside the universe (design, withdrawn, pre-1976) giving anyone credit;
- a person listed twice on one patent counted as two patents.
"""

import duckdb
import pytest

import build_dominance

# (patent_id, patent_type, withdrawn, filing date, CPC group with sequence 0 or None, inventors)
WORLD = [
    ("P1", "utility", "0", "1995-03-01", "A01B1/02", ["I1", "I2"]),
    ("P2", "utility", "0", "1996-03-01", "A01B1/04", ["I1"]),
    ("P3", "design", "0", "1996-03-01", "A01B1/02", ["I3"]),     # not an invention
    ("P4", "utility", "1", "1996-03-01", "A01B1/02", ["I3"]),     # withdrawn
    ("P5", "utility", "0", "1975-06-01", "A01B1/02", ["I3"]),     # filed before 1976
    ("P6", "utility", "0", "1997-03-01", "A01B3/02", ["I4", "I5"]),  # a tie at the top
    ("P7", "utility", "0", "1997-03-01", "A01B1/02", []),         # no inventor record
    ("P8", "utility", "0", "1997-03-01", "A01B1/02", ["I2", "I2", "I6"]),  # I2 listed twice
    ("P9", "utility", "0", "1997-03-01", None, ["I7"]),           # no primary CPC code
]


def write_tsv(path, header, rows):
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def built(tmp_path):
    """The made-up extract written as PatentsView .tsv files, then the real build run on it."""
    data = tmp_path / "data"
    data.mkdir()
    write_tsv(data / "g_patent.tsv", ["patent_id", "patent_type", "patent_date", "withdrawn"],
              [(p, kind, "2000-01-01", withdrawn) for p, kind, withdrawn, _, _, _ in WORLD])
    write_tsv(data / "g_application.tsv", ["patent_id", "filing_date"],
              [(p, filed) for p, _, _, filed, _, _ in WORLD])
    write_tsv(data / "g_inventor_disambiguated.tsv",
              ["patent_id", "inventor_sequence", "inventor_id"],
              [(p, str(s), i) for p, *_, team in WORLD for s, i in enumerate(team)])
    # P9 has only a secondary code (sequence 1), so it has no field.
    write_tsv(data / "g_cpc_current.tsv", ["patent_id", "cpc_sequence", "cpc_group"],
              [(p, "0", group) if group else (p, "1", "A01B9/02")
               for p, _, _, _, group, _ in WORLD])
    build_dominance.run(data, tmp_path / "out")
    return tmp_path / "out"


def rows(out, sql):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW inventors AS SELECT * FROM '{(out / 'inventor_field_years.parquet').as_posix()}'")
    con.execute(f"CREATE VIEW fields AS SELECT * FROM '{(out / 'field_years.parquet').as_posix()}'")
    return con.sql(sql).fetchall()


def test_shares_sum_to_one_in_every_field_year(built):
    totals = rows(built, "SELECT sum(share) FROM inventors GROUP BY field_id, year")
    assert max(abs(total - 1.0) for (total,) in totals) < 1e-12


def test_a_patent_enters_the_five_windows_from_its_filing_year_and_no_earlier_one(built):
    """I4's only patent is filed in 1997, so she appears in the windows ending 1997 to 2001."""
    years = rows(built, "SELECT year FROM inventors WHERE inventor_id = 'I4' ORDER BY year")
    assert [y for (y,) in years] == [1997, 1998, 1999, 2000, 2001]


def test_two_inventors_level_at_the_top_share_rank_one_and_have_margin_one(built):
    """If two people are level, neither towers over anyone: both rank 1, margin exactly 1."""
    result = rows(built, """SELECT inventor_id, rank, n_tied_at_rank, margin_uncapped
                            FROM inventors WHERE field_id = 'A01B3/00' AND year = 1997
                            ORDER BY inventor_id""")
    assert result == [("I4", 1, 2, 1.0), ("I5", 1, 2, 1.0)]


def test_patents_outside_the_universe_give_no_one_credit(built):
    """Design, withdrawn, pre-1976 and field-less patents are out; a patent with no inventor
    still counts in its field's patent total."""
    assert rows(built, "SELECT count(*) FROM inventors WHERE inventor_id IN ('I3', 'I7')") == [(0,)]
    # Window 1997 of A01B1/00 holds P1, P2, P7 and P8.
    assert rows(built, """SELECT field_patents_in_window FROM fields
                          WHERE field_id = 'A01B1/00' AND year = 1997""") == [(4,)]


def test_a_person_listed_twice_on_a_patent_holds_two_shares_of_it_and_one_patent(built):
    """I2: half of P1, and two thirds of P8 (named twice on a team of three)."""
    result = rows(built, """SELECT inventor_weight_in_window, inventor_patents_in_window
                            FROM inventors WHERE inventor_id = 'I2' AND field_id = 'A01B1/00'
                              AND year = 1997""")
    assert result == [(round(0.5 + 2 / 3, 9), 2)]


def test_the_sum_to_one_check_catches_a_head_count_denominator(tmp_path, built, monkeypatch):
    """
    The share's denominator is one word away from the field's patent head count, and both
    run. Put the wrong word into the real query and the build must stop.
    """
    right = "/ sum(inventor_weight_in_window)"
    assert right in build_dominance.RANKED_SQL
    monkeypatch.setattr(build_dominance, "RANKED_SQL", build_dominance.RANKED_SQL.replace(
        right, "/ sum(inventor_patents_in_window)"))
    with pytest.raises(ValueError, match="shares sum to 1"):
        build_dominance.run(tmp_path / "data", tmp_path / "out_broken")
