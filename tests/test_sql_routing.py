"""Sawal -> metric aur saal. Dev aur test sets mein pakri gayi ghaltiyan yahan pakki ho jati hain."""
import re

import pytest

from financials_db import DESCRIPTIVE, detect_metrics, fmt, question_years


@pytest.mark.parametrize("question, metric", [
    ("What was Apple's total revenue in 2021?", "revenue"),
    ("walmart net sales 2020", "revenue"),
    ("How much profit did Southwest make in 2018?", "net_income"),
    ("Intel total assets 2019", "total_assets"),
    ("how much were Boeing's assets in 2018", "total_assets"),
    ("Microsoft liabilities 2022", "total_liabilities"),
    ("Nike cash from operations 2017", "operating_cash_flow"),
    ("Chevron capital spending 2016", "capex"),
    ("Apple diluted EPS 2021", "eps_diluted"),
    ("How much cash did Adobe have in 2017?", "cash"),
    ("Intel R&D 2018", "rnd"),
])
def test_detects_metric(question, metric):
    assert metric in detect_metrics(question)


@pytest.mark.parametrize("question, not_metric", [
    ("Apple gross profit 2021", "net_income"),
    ("Apple operating profit 2021", "net_income"),
    ("Intel current assets 2019", "total_assets"),
    ("Microsoft lease liabilities 2022", "total_liabilities"),
    ("Nike free cash flow 2017", "cash"),
    ("PayPal cash dividends 2022", "cash"),
])
def test_does_not_confuse_related_metrics(question, not_metric):
    assert not_metric not in detect_metrics(question)


def test_non_number_question_has_no_metric():
    assert detect_metrics("Who is the CEO of Meta in 2024?") == []


def test_descriptive_questions_are_excluded():
    assert DESCRIPTIVE.search("What risks did Chevron describe in 2016?")
    assert DESCRIPTIVE.search("Explain Apple's revenue recognition policy")
    assert not DESCRIPTIVE.search("What was Apple's revenue in 2021?")


def test_range_expands_to_every_year():
    assert question_years("Intel revenue 2019-2022", {2019, 2022}) == {2019, 2020, 2021, 2022}
    assert question_years("cash from 2018 through 2021", set()) == {2018, 2019, 2020, 2021}


def test_span_keeps_only_the_ends_unless_each_year_is_asked():
    assert question_years("change in revenue from 2019 to 2022", {2019, 2022}) == {2019, 2022}
    assert question_years("revenue each year from 2019 to 2022", {2019, 2022}) == {2019, 2020, 2021, 2022}


def test_backwards_or_very_long_range_is_ignored():
    assert question_years("2022-2019", set()) == set()
    assert question_years("1990-2020", set()) == set()


@pytest.mark.parametrize("value, metric, text", [
    (394_328_000_000, "revenue", "$394.33 billion"),
    (812_400_000, "revenue", "$812.40 million"),
    (-1_200_000, "net_income", "-$1.20 million"),
    (6.11, "eps_diluted", "$6.11 per share"),
    (-0.5, "eps_diluted", "-$0.50 per share"),
    (None, "revenue", "n/a"),
])
def test_money_formatting(value, metric, text):
    assert fmt(value, metric) == text


def test_retriever_year_pattern_handles_commas():
    # R23: "2019," pehle ignore ho jata tha; "12,020" ya "$2,019" saal nahi
    retriever = pytest.importorskip("retriever")
    yp = retriever.Retriever.YEAR_PATTERN
    assert {int(y) for y in yp.findall("Apple revenue in 2019, and 2020")} == {2019, 2020}
    assert yp.findall("it had 12,020 employees") == []
    assert yp.findall("costs of $2,019 million") == []
    assert not re.search(yp, "20190")