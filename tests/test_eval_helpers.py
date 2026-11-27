"""Imtihan ke auzaar khud sahi hon: confidence interval, number check, aur jawab ki category."""
import pytest

from eval_sql import wilson


def test_wilson_interval_matches_reported_results():
    lo, hi = wilson(149, 150)                     # SQL held-out routing: 99.3% (96.3-99.9)
    assert round(100 * lo, 1) == 96.3 and round(100 * hi, 1) == 99.9


def test_wilson_handles_edges():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(50, 50)
    assert hi == 1.0 and lo > 0.9


E = pytest.importorskip("eval_answers")          # needs the retriever's libraries (faiss, scipy)


def row(answer, question="How much did X earn in 2021?", type_="number"):
    return {"type": type_, "answer": answer, "question": question}


def test_number_in_answer_ignores_formatting():
    assert E.number_in_answer(row("1,234.5"), "Revenue was $1234.5 million [S1].")


def test_number_in_answer_accepts_any_gold_alternative():
    assert E.number_in_answer(row("394,328|394.3"), "It was $394.3 billion [D1].")


def test_number_in_answer_does_not_count_the_year_from_the_question():
    assert not E.number_in_answer(row("812"), "In 2021 the figure is not stated.")


def test_number_check_only_applies_to_number_questions():
    assert E.number_in_answer(row("Tim Cook", type_="text"), "Tim Cook") is None


def test_oversized_prompt_is_its_own_category():
    assert E.judge(None, {}, {"error": "too_large"}) == ("too_large", None)
    assert E.judge(None, {}, {"error": "no_model"}) == ("no_model", None)
    assert "too_large" in E.SKIP and "no_model" in E.SKIP