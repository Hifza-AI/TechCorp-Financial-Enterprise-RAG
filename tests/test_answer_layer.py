"""Answer layer: citations saaf karna, aur Groq ki limits par sahi rawayya (koi asli API call nahi)."""
import pytest

import answer as A


@pytest.mark.parametrize("raw, clean", [
    ("Revenue rose 【S1†L1-L9】.", "Revenue rose [S1]."),
    ("Revenue rose 【S3】.", "Revenue rose [S3]."),
    ("Revenue rose (S2).", "Revenue rose [S2]."),
    ("Both grew [S1, S2; D1].", "Both grew [S1][S2][D1]."),
    ("2019–2021 — up", "2019-2021 - up"),
])
def test_clean_answer_normalises_citations(raw, clean):
    assert A.clean_answer(raw) == clean


def test_cited_sources_keeps_only_cited_tags():
    srcs = [{"tag": "D1"}, {"tag": "S1"}, {"tag": "S2"}]
    assert [s["tag"] for s in A.cited_sources("Yes [D1] and [S2].", srcs)] == ["D1", "S2"]


def test_groq_error_message_is_extracted():
    body = '{"error": {"message": "Request too large ... Limit 8000, Requested 8958"}}'
    assert A._groq_msg(body).startswith("Request too large")
    assert A._groq_msg("plain text") == "plain text"


def test_answer_budget_shrinks_when_request_is_slightly_too_large():
    A._GROQ_MAX.clear()
    assert A._shrink_on_too_large("m", "p", "Limit 8000, Requested 8958")
    assert A._GROQ_MAX[("m", hash("p"))] == 2048 - 958 - 150


def test_answer_budget_is_not_shrunk_below_a_useful_size():
    A._GROQ_MAX.clear()
    assert not A._shrink_on_too_large("m", "p", "Limit 8000, Requested 9900")
    assert not A._shrink_on_too_large("m", "p", "some other error")


class FakeClient:
    groq_key = "test"
    gemini = None


def run_chain(monkeypatch, behaviour, models=("groq:a", "groq:b")):
    """behaviour: model -> list of results/exceptions, consumed one call at a time."""
    calls = []

    def fake_call(key, model, prompt, system):
        calls.append(model)
        out = behaviour[model].pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(A, "call_groq", fake_call)
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    chain = A.ModelChain(list(models))
    return chain, calls, lambda: chain.generate(FakeClient(), "prompt", log=lambda m: None)


def test_prompt_too_large_for_every_model_raises_too_large(monkeypatch):
    big = A.Busy("groq:x", 60, "Request too large for model")
    _, calls, gen = run_chain(monkeypatch, {"a": [big], "b": [big]})
    with pytest.raises(A.TooLarge):
        gen()
    assert calls == ["a", "b"]          # no waiting and retrying: waiting can never help


def test_daily_limit_moves_to_next_model_and_remembers(monkeypatch):
    chain, calls, gen = run_chain(monkeypatch, {"a": [A.DailyQuota("a")], "b": ["ok", "ok2"]})
    assert gen() == ("groq:b", "ok")
    assert gen() == ("groq:b", "ok2")   # dead model is not tried again
    assert calls == ["a", "b", "b"]


def test_minute_limit_waits_and_retries(monkeypatch):
    busy = A.Busy("groq:a", 5, "Rate limit reached ... Please try again in 4s")
    _, calls, gen = run_chain(monkeypatch, {"a": [busy, "ok"]}, models=("groq:a",))
    assert gen() == ("groq:a", "ok")
    assert calls == ["a", "a"]