"""[D1] source ek chhote, banaye hue database par: restatement, saal, companies ka farq, ghalat status."""
import sqlite3

import pytest

from financials_db import SCHEMA, FinancialData

ROWS = [
    # company, fy, metric, value, status, page, prior_value
    ("Boeing", 2017, "revenue", 93_392e6, "sec_verified", "52", None),
    ("Boeing", 2018, "revenue", 101_127e6, "sec_verified", "60", 94_005e6),   # FY2017 restated (ASC 606)
    ("Boeing", 2019, "revenue", 76_559e6, "sec_only", None, 101_127e6),
    ("Boeing", 2018, "total_assets", 3_680e6, "pdf_suspect", "70", None),     # segment table: must not be served
    ("Intel", 2018, "revenue", 70_848e6, "sec_verified", "40", 62_761e6),
]


@pytest.fixture()
def fin(tmp_path):
    db = tmp_path / "facts.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO financials(company, fiscal_year, metric, value, status, page, prior_value, source)"
        " VALUES (?,?,?,?,?,?,?,'sec')", ROWS)
    con.commit()
    con.close()
    return FinancialData(db)


def test_missing_database_turns_the_feature_off(tmp_path):
    fd = FinancialData(tmp_path / "nope.db")
    assert not fd.available
    assert fd.source("Boeing revenue 2018", {"Boeing"}, {2018}) is None


def test_single_year_brings_prior_year_and_change(fin):
    src = fin.source("What was Boeing's revenue in 2018?", {"Boeing"}, {2018})
    assert src["tag"] == "D1" and src["type"] == "structured"
    assert {r["fiscal_year"] for r in src["rows"]} == {2017, 2018}
    # change is against the restated comparative in the same filing (+7.6%), not the original 93.4B (+8.3%)
    assert "+7.6%" in src["text"]
    assert "restated" in src["text"]


def test_no_year_gives_the_latest_three(fin):
    src = fin.source("Boeing revenue", {"Boeing"}, set())
    assert [r["fiscal_year"] for r in src["rows"]] == [2017, 2018, 2019]


def test_descriptive_question_gets_no_numbers(fin):
    assert fin.source("Why did Boeing's revenue fall in 2019?", {"Boeing"}, {2019}) is None


def test_suspect_rows_are_never_served(fin):
    assert fin.source("Boeing total assets 2018", {"Boeing"}, {2018}) is None


def test_two_companies_get_a_precomputed_difference(fin):
    src = fin.source("Boeing vs Intel revenue 2018", {"Boeing", "Intel"}, {2018})
    assert "Difference FY2018" in src["text"]
    assert "Boeing is higher than Intel" in src["text"]


def test_question_without_a_metric_gets_nothing(fin):
    assert fin.source("Who is Boeing's CEO in 2018?", {"Boeing"}, {2018}) is None