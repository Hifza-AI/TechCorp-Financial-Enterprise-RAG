"""
api.py  -  RAG backend ko HTTP service ki shakal: wahi answer layer jo ask/answer.py chalate hain.

Chalao (repo root se):
    python -m uvicorn api:app --app-dir STAGE_1 --port 8000
Phir:
    GET  http://localhost:8000/health
    POST http://localhost:8000/ask   {"question": "What was Apple's total revenue in fiscal 2021?"}
    Docs (browser mein try karo):  http://localhost:8000/docs

Settings (.env ya environment):  LLM_MODELS, STRUCTURED_DATA, FIN_DB, VECTOR_STORE (default STAGE_1/vector_store),
ANSWER_CACHE (default STAGE_1/answer_cache.json), REQUEST_LOG (default STAGE_1/logs/requests.jsonl).
Har sawal ka ek JSON line log: waqt, model, cache, error, prompt ka size -- latency aur kharche ke hisaab ke liye.
"""
import hashlib
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer as A  # noqa: E402

log = logging.getLogger("rag.api")
MAX_QUESTION = 500                  # is se lamba sawal galti ya ghalat istemal hai
SOURCE_TEXT_LIMIT = 1500            # response mein har source ka itna text (poora chunk UI ke liye kaafi)
ERROR_STATUS = {"no_model": 503, "network": 503, "too_large": 503}
ERROR_TEXT = {
    "no_model": "All language models are at their rate limit. Try again in a few minutes.",
    "network": "Could not reach the language model provider.",
    "too_large": "This question needs more context than the model's per-minute limit allows.",
}


# ---------------------------------------------------------------- request / response shapes
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_QUESTION,
                          examples=["What was Apple's total revenue in fiscal 2021?"])
    k: int = Field(5, ge=1, le=10, description="How many retrieved chunks the model sees")


class Source(BaseModel):
    tag: str
    company: str
    fiscal_year: Optional[int] = None
    type: str
    section: str
    pages: List[int] = []
    text: str
    cited: bool
    rows: Optional[List[dict]] = Field(None, description="Structured source only: each figure with its SEC "
                                                        "status, PDF page and filing link")


class AskResponse(BaseModel):
    question: str
    answer: str
    not_stated: bool
    model: Optional[str]
    cached: bool
    structured_data: bool
    seconds: float
    sources: List[Source]


class Health(BaseModel):
    status: str
    models: List[str]
    structured_data: bool
    prompt_version: str


# ---------------------------------------------------------------- backend
class Backend:
    """Retriever, model chain aur cache EK dafa load; har sawal answer.answer_one se (CLI wala hi rasta)."""

    def __init__(self, retriever, client, chain, cache, request_log=None):
        self.retriever, self.client, self.chain, self.cache = retriever, client, chain, cache
        self.request_log = Path(request_log) if request_log else None
        self.lock = threading.Lock()     # cache file aur model chain ek waqt mein ek sawal

    @classmethod
    def from_env(cls):
        A.load_env()
        from retriever import Retriever
        client = A.make_client()
        return cls(
            retriever=Retriever(os.environ.get("VECTOR_STORE", str(HERE / "vector_store"))),
            client=client,
            chain=A.build_chain(client),
            cache=A.Cache(os.environ.get("ANSWER_CACHE", str(HERE / "answer_cache.json"))),
            request_log=os.environ.get("REQUEST_LOG", str(HERE / "logs" / "requests.jsonl")),
        )

    def ask(self, question, k=5):
        with self.lock:
            entry = A.answer_one(self.retriever, self.client, self.cache, self.chain, question, k)
        self._log(question, entry)
        return entry

    def _log(self, question, entry):
        if not self.request_log:
            return
        rec = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question_id": hashlib.sha256(question.encode("utf-8")).hexdigest()[:12],
            "question": question,
            "model": entry.get("model"),
            "cached": entry.get("cached", False),
            "error": entry.get("error"),
            "not_stated": entry.get("not_stated", False),
            "structured": entry.get("structured", False),
            "sources": len(entry.get("sources") or []),
            "prompt_chars": sum(len(s.get("text", "")) for s in entry.get("sources") or []),
            "answer_chars": len(entry.get("answer") or ""),
            "seconds": entry.get("seconds"),
        }
        try:
            self.request_log.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:                   # log na likh sake to jawab phir bhi do
            log.warning("request log not written: %s", e)


def to_response(entry):
    cited = {c["tag"] for c in entry.get("citations") or []}
    sources = []
    for s in entry.get("sources") or []:
        year = s.get("year")
        sources.append(Source(
            tag=s["tag"], company=s.get("company", ""),
            fiscal_year=int(year) if isinstance(year, int) or (isinstance(year, str) and year.isdigit()) else None,
            type=s.get("type", ""), section=s.get("section", ""),
            pages=[int(p) for p in s.get("pages") or [] if str(p).isdigit()],
            text=(s.get("text") or "")[:SOURCE_TEXT_LIMIT], cited=s["tag"] in cited, rows=s.get("rows")))
    return AskResponse(
        question=entry["question"], answer=entry["answer"], not_stated=bool(entry.get("not_stated")),
        model=entry.get("model"), cached=bool(entry.get("cached")), structured_data=bool(entry.get("structured")),
        seconds=float(entry.get("seconds") or 0.0), sources=sources)


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    if getattr(app.state, "backend", None) is None:      # tests apna (nakli) backend pehle se rakhte hain
        app.state.backend = Backend.from_env()
    yield


app = FastAPI(title="TechCorp Financial RAG API", version="1.0",
              description="Questions over 500 SEC 10-K filings, answered with page-level citations.",
              lifespan=lifespan)
app.state.backend = None


@app.get("/health", response_model=Health)
def health(request: Request):
    b = request.app.state.backend
    return Health(status="ok", models=list(b.chain.models), structured_data=A.financial_data() is not None,
                  prompt_version=A.PROMPT_VERSION)


@app.post("/ask", response_model=AskResponse, responses={503: {"description": "No model available right now"}})
def ask(req: AskRequest, request: Request):
    question = " ".join(req.question.split())
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is empty.")
    entry = request.app.state.backend.ask(question, req.k)
    err = entry.get("error")
    if err in ERROR_STATUS:
        return JSONResponse(status_code=ERROR_STATUS[err], headers={"Retry-After": "60"},
                            content={"error": err, "detail": ERROR_TEXT[err]})
    return to_response(entry)