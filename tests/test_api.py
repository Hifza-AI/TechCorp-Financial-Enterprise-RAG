"""HTTP service: nakli backend ke saath (koi model, retriever ya API key nahi)."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import answer as A  # noqa: E402
import api  # noqa: E402


class FakeRetriever:
    YEAR_PATTERN = A.re.compile(r"(?:19|20)\d{2}")

    def __init__(self, matched=True):
        self.matched = matched

    def search(self, question, top_k=5):
        if not self.matched:
            return {"matched": False, "reason": "company not found", "results": []}
        return {"matched": True, "results": [{
            "company_display": "Apple", "company": "Apple", "year": 2021, "section_path": "Item 7 > Revenue",
            "page_numbers": [21, 22], "chunk_type": "table", "text": "Total net sales 365,817"}]}

    def _detect_companies(self, question):
        return {"Apple"}


class FakeChain:
    def __init__(self, outcome):
        self.models = ["groq:fake"]
        self.outcome = outcome
        self.calls = 0

    def generate(self, client, prompt, log=print, **kw):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return "groq:fake", self.outcome


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "financial_data", lambda: None)       # no facts.db in tests

    def make(outcome="Apple's total net sales were $365,817 million [S1].", matched=True):
        backend = api.Backend(FakeRetriever(matched), None, FakeChain(outcome), A.Cache(tmp_path / "cache.json"),
                              request_log=tmp_path / "requests.jsonl")
        api.app.state.backend = backend
        return TestClient(api.app), backend, tmp_path / "requests.jsonl"
    yield make
    api.app.state.backend = None


def test_health_reports_models(make_client):
    client, _, _ = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["models"] == ["groq:fake"]


def test_ask_returns_answer_with_cited_sources(make_client):
    client, _, _ = make_client()
    r = client.post("/ask", json={"question": "What was Apple's total revenue in fiscal 2021?"})
    assert r.status_code == 200
    body = r.json()
    assert "[S1]" in body["answer"] and body["model"] == "groq:fake"
    src = body["sources"][0]
    assert src["company"] == "Apple" and src["fiscal_year"] == 2021 and src["pages"] == [21, 22] and src["cited"]


def test_second_identical_question_is_served_from_cache(make_client):
    client, backend, _ = make_client()
    q = {"question": "What was Apple's total revenue in fiscal 2021?"}
    client.post("/ask", json=q)
    r = client.post("/ask", json=q)
    assert r.json()["cached"] is True
    assert backend.chain.calls == 1


def test_unknown_company_is_not_stated_without_calling_a_model(make_client):
    client, backend, _ = make_client(matched=False)
    r = client.post("/ask", json={"question": "What was Foo Corp's revenue in 2021?"})
    assert r.status_code == 200 and r.json()["not_stated"] is True
    assert backend.chain.calls == 0


@pytest.mark.parametrize("exc, code", [(A.DailyQuota("all"), "no_model"), (A.TooLarge("m"), "too_large"),
                                       (A.NetworkDown("x"), "network")])
def test_model_problems_return_503_with_a_reason(make_client, exc, code):
    client, _, _ = make_client(outcome=exc)
    r = client.post("/ask", json={"question": "What was Apple's total revenue in fiscal 2021?"})
    assert r.status_code == 503 and r.json()["error"] == code
    assert r.headers["Retry-After"] == "60"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 501}, {"question": "ok?", "k": 50}, {}])
def test_bad_requests_are_rejected(make_client, payload):
    client, backend, _ = make_client()
    assert client.post("/ask", json=payload).status_code == 422
    assert backend.chain.calls == 0


def test_each_question_is_logged(make_client):
    client, _, log = make_client()
    client.post("/ask", json={"question": "What was Apple's total revenue in fiscal 2021?"})
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["model"] == "groq:fake" and rec["cached"] is False and rec["sources"] == 1
    assert len(rec["question_id"]) == 12 and rec["seconds"] is not None