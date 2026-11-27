"""Fiscal year ke qaide: spot-check mein pakre gaye Dell aur Franklin wale masle dobara na aayein."""
from datetime import date

from validate_xbrl import fy_of, matches, pick_accn


def test_calendar_year_end():
    assert fy_of(date(2021, 12, 31)) == 2021


def test_52_53_week_year_ending_in_early_january_belongs_to_previous_year():
    # Franklin Electric fiscal 2015 ended on January 2, 2016
    assert fy_of(date(2016, 1, 2)) == 2015
    assert fy_of(date(2016, 1, 10)) == 2015


def test_late_january_and_february_year_ends_keep_their_end_year():
    # Walmart and Dell end their year at the end of January / start of February
    assert fy_of(date(2020, 1, 31)) == 2020
    assert fy_of(date(2019, 2, 1)) == 2019
    assert fy_of(date(2016, 1, 11)) == 2016


# accession tuples: (accn, form, filed, period_end)
ACCNS = [
    ("A-2018", "10-K", "2018-03-28", date(2018, 2, 2)),
    ("A-2020", "10-K", "2020-03-27", date(2020, 1, 31)),
    ("A-2020-amend", "10-K/A", "2020-04-10", date(2020, 1, 31)),
]


def test_pick_accn_returns_that_years_filing():
    assert pick_accn(ACCNS, 2018)[0] == "A-2018"


def test_pick_accn_never_borrows_the_next_years_filing():
    # Dell FY2019: its own filing is missing -> None, not the FY2020 filing
    assert pick_accn(ACCNS, 2019) is None


def test_pick_accn_prefers_original_10k_over_amendment():
    assert pick_accn(ACCNS, 2020)[0] == "A-2020"


def test_pick_accn_applies_company_offset():
    assert pick_accn(ACCNS, 2017, off=1)[0] == "A-2018"


def test_matches_detects_scale():
    assert matches(394_328, 394_328_000_000, "revenue") == 1e6
    assert matches(394.3, 394_300_000_000, "revenue") == 1e9


def test_matches_tolerates_display_rounding_and_sign():
    assert matches(-695, 695_400_000, "capex") == 1e6


def test_matches_rejects_a_different_number():
    assert matches(400_000, 394_328_000_000, "revenue") is None


def test_matches_eps_is_exact_to_the_cent():
    assert matches(6.11, 6.11, "eps_diluted") == 1.0
    assert matches(6.13, 6.11, "eps_diluted") is None