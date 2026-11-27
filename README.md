# TechCorp Financial Enterprise RAG

[![tests](https://github.com/Hifza-AI/TechCorp-Financial-Enterprise-RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/Hifza-AI/TechCorp-Financial-Enterprise-RAG/actions/workflows/tests.yml)

A local, verifiable Retrieval-Augmented Generation (RAG) knowledge base over **500 SEC 10-K annual reports** (50 US public companies × 10 fiscal years). Every answer is traceable to a company, fiscal year, section and page number in the original filing.

**Final Year Project** — BS INFORMATION TECHNOLOGY, Emerson University Multan
**Supervisor:** Engr. Dr. Muhammad Naveed Yasir
**Team:** Hifza Riaz (COSC232112112), Saman Saif (COSC232112102)

-----

## Why this project

Financial filings are long (100–300 pages each), heavily tabular, and formatted differently by every company. Generic RAG tools split them into fixed-size text windows, which breaks tables, loses section context, and makes answers impossible to verify. This project builds the document-understanding layer first — headings, tables, hierarchy — and only then retrieves and answers, so that every result can be checked against the source page.

## What is built so far

| # | Module | Status | Files |
|---|--------|--------|-------|
| 1 | PDF extraction with font/position metadata | ✅ Complete | `pdf_extractor.py` |
| 2 | Text cleaning + cleaning rules | ✅ Complete | `text_cleaner.py`, `cleaning_rules.py` |
| 3 | Font-based heading detection (company-agnostic) | ✅ Complete | `heading_detector.py` |
| 4 | Table detection + parsing | ✅ Complete | `table_analyzer.py`, `table_parser.py` |
| 5 | Paragraph parsing + boilerplate removal | ✅ Complete | `paragraph_parser.py`, `fix_paragraphs.py` |
| 6 | Document hierarchy (nested section tree) | ✅ Complete | `hierarchy_builder.py` |
| 7 | Chunking with citation metadata | ✅ Complete | `chunk_builder.py` |
| 8 | Embeddings + FAISS index | ✅ Complete | `build_embeddings.py` |
| 9 | Hybrid retriever + evaluation suite | ✅ Complete, frozen (v2.4.1) | `retriever.py`, `eval_golden.py`, `ask.py` |
| 10 | SQL track: PDF facts, SEC XBRL, validation, serving layer, routing | ✅ Complete | `facts_db.py`, `metrics_db.py`, `xbrl_db.py`, `validate_xbrl.py`, `financials_db.py`, `eval_sql.py`, `spotcheck.py` |
| 11 | Answer layer: Groq / Gemini model chain, citations, "not stated" guard, SQL source | 🔄 Final evaluation | `answer.py`, `eval_answers.py`, `judge_calibration.py`, `compare_runs.py` |
| 12 | HTTP API: `/health`, `/ask`, one JSON log line per request | ✅ Complete | `api.py` |
| 13 | Automated tests (78, offline) + GitHub Actions CI | ✅ Complete | `tests/`, `.github/workflows/tests.yml` |
| 14 | Streamlit app | ⬜ Planned | — |
| 15 | Deployment (Docker) | ⬜ Planned | — |

## Dataset

- **Source:** SEC EDGAR — original 10-K annual report PDFs (no synthetic data)
- **Scope:** 50 companies × 10 fiscal years = **500 filings**
- **Output:** **40,027** retrieval chunks (text + whole-table chunks), **5,313** parsed tables → **147,640** structured facts
- Year labels are **fiscal** years as stated in each filing (e.g. a company whose fiscal year ends in January files FY2024 in early 2025).

## Pipeline

```
PDF  →  extract (text + font size + bold + bbox)
     →  clean (URLs, timestamps, browser-print footers, unicode)
     →  detect headings (relative font size, bold/italic, line shape — no hardcoded "Item"/"Note")
     →  detect + parse tables (x/y alignment → rows/columns → {label, values})
     →  parse paragraphs (frequency-based boilerplate removal)
     →  build hierarchy (nested Item → Note → subsection tree; tables attached by position)
     →  build chunks (≤1200 chars, 150 overlap; one table = one chunk; section path + pages kept)
     →  embed (bge-small-en-v1.5, 384-d) → FAISS index
```

Every rule in the pipeline was added after a confirmed failure on a real filing; the file comments record which company and filing it was verified on (Apple, Nike, Intel, Costco, Dell, Netflix, PayPal, CVS, Nvidia, Chipotle, Workday, Caterpillar and others).

## Retrieval

`retriever.py` combines:

- **FAISS** dense search (semantic match)
- **BM25** sparse search (exact tokens — names, figures, tickers)
- **Reciprocal Rank Fusion** to merge both rankings
- Company / fiscal-year filters, and v2.3 query features: section intent, typo repair, ticker names, synonyms, Roman-Urdu terms

Everything runs locally. No API calls, no data leaves the machine.

## Evaluation

A 100-question test set (`STAGE_1/test_questions.csv`) was written and **frozen in git before** the retriever was run on it. Each question carries the expected answer and page; checking is automatic.

| Test set | Questions | Correct chunk in top-5 | Hit@1 | MRR |
|----------|-----------|------------------------|-------|-----|
| Fair test — v2.2 (before fixes, unbiased) | 100 | **80** | 52 | 0.63 |
| Fair test — v2.3 (after fixes) | 100 | **90** | 56 | 0.70 |
| Golden set (hard questions) | 56 | 85.4% | — | — |
| Large eval set | 450 | 96.6% | — | — |
| Out-of-scope (companies not in the dataset) | 3 | 3/3 correctly return no match | | |

Per-feature gains from ablation (v2.3): intent +5, typo +3, tickers +2, words +1, roman 0.

> The v2.3 fixes were designed from this set's misses, so 90/100 is an in-sample figure. The unbiased number is 80/100. The retriever is frozen at v2.4.1; its final figure comes from the held-out set below.

### SQL track (number questions)

| Check | Result |
|-------|--------|
| PDF-extracted metrics vs SEC XBRL | **97.9%** (4,250 / 4,339) |
| Serving values checked by hand against the PDFs | **59/60** after fixes (55/60 at audit) |
| Question → right company, metric, year and SEC value (fresh held-out set) | **99.3%** (149/150, 95% CI 96.3–99.9) |
| Non-number questions given structured data by mistake | **0/50** |

The manual audit found two fiscal-year mapping bugs that no automatic test could see, because the test gold used the same mapping; both were fixed and the gold corrected with a logged change.

### Answer layer (LLM)

| Measure (dev set, 95 of 100 questions) | Result |
|-------|--------|
| Correct, human labels | **92.6%** |
| Correct, LLM judge (Qwen3) | 88.4% |
| Judge–human agreement | 90.5% (Cohen's κ 0.46) |
| Faithfulness (no unsupported claim), human labels | **91.6%** |

The judge was checked blind against human labels. It is stricter on completeness but misses small unsupported additions, so human-labelled faithfulness is reported.

**Does the SQL source help?** Same model (GPT-OSS-120B), same judge, same 99 questions, only the [D1] source switched on:

| Dev set, 99 questions | SQL off | SQL on |
|---|---|---|
| Correct (LLM judge) | 87 (87.9%) | **88 (88.9%)** |
| Faithfulness (LLM judge) | 94 (94.9%) | **95 (96.0%)** |
| The 12 questions that receive [D1] | 10 correct | **11 correct** |
| Answers that got worse | | **0** |

One answer was fixed (ServiceNow 2016 revenue) and none got worse. The gain is small because only 12 dev questions are number questions and 10 of them were already right; the held-out set has 21 number and 10 comparison questions. One question (T96) is excluded from both runs because its prompt is larger than the free tier's per-minute limit (A6).

Still to run: prompt v3 vs v4 on the dev set, then one run on **60 held-out questions written by hand** from random 10-K pages (never used for tuning).

Known issues are documented in [`STAGE_1/known_limitations.md`](STAGE_1/known_limitations.md) (retrieval R1–R23, SQL track F1–F6, answer layer A1–A5).

## How to run

```bash
# 1. Ask questions interactively (model loads once)
python STAGE_1/ask.py
#   Sawal> Who is the CEO of Roper Technologies?
#   Sawal> 52 | What was Nike's net income in fiscal 2021?   # "52 |" = also check that page 52 is retrieved

# 2. Run the 100-question evaluation
python STAGE_1/retriever.py --test

# 3. Per-feature ablation on the golden set
python STAGE_1/eval_golden.py --ablation

# 4. Build the facts database from parsed tables
python STAGE_1/facts_db.py

# 5. Serve the answer layer over HTTP (API docs at http://localhost:8000/docs)
python -m uvicorn api:app --app-dir STAGE_1 --port 8000

# 6. Run the tests (offline: no API keys, vector store or database needed)
python -m pip install -r requirements-test.txt
python -m pytest
```

API keys go in a `.env` file at the repo root (`GROQ_API_KEY`, `GEMINI_API_KEY`), which is git-ignored.

To rebuild from PDFs, run the pipeline scripts in the order listed in the table above (`pdf_extractor.py` → … → `build_embeddings.py`). Each script reads the previous stage's folder under `STAGE_1/` and writes its own.

### Requirements

Python 3.12. Versions are pinned to the ones the reported results were produced with.

```bash
python -m pip install -r requirements.txt
```

## Cost and rate limits

Measured on the Groq usage dashboard, not estimated from a price list:

| Item | Figure |
|---|---|
| One answer (GPT-OSS-120B, about 5,000 input tokens) | about **$0.001** |
| Largest prompt seen (about 9,000 input tokens, full 2,048-token answer budget) | about $0.003 |
| A full day of evaluation (about 200 model calls, answers plus judge) | **$0.18** at paid rates |

Most tokens are input (sources), not output: on 26 Nov the answer model used 201.7K input and 5.1K output tokens. Fewer or shorter sources would cut cost the most.

The free tier allows 8,000 tokens per minute and 200,000 per day on GPT-OSS-120B (about 45 answers a day), and the daily budget is a rolling 24-hour window, not a midnight reset. The code handles this instead of failing:

- **Minute limit:** wait for the time Groq gives, then retry.
- **Daily limit:** move to the next model in the chain (Gemini, other Groq models) and remember the exhausted one.
- **Prompt over the limit:** shrink the answer budget once; if the prompt alone is too large, report `too_large` instead of waiting.
- **Cache:** a repeated question costs nothing.
- **Evaluation:** refuses to run without an explicit model and judge, so a run never mixes models silently.

For a public deployment, the paid tier (no daily cap) with a spend limit in the Groq console bounds the monthly bill.

## Repository layout

```
STAGE_1/
├── data/pdfs/<Company>/        original 10-K PDFs
├── extracted/ cleaned/ heading_detection/ table_analysis/
├── parsed_tables/ paragraphs/ paragraphs_fixed/ hierarchy/
├── chunks/<Company>/           final chunks (JSON)
├── eval/                       test results and ablation outputs
├── logs/                       API request log (git-ignored)
├── test_questions.csv          frozen 100-question fair test
├── known_limitations.md        documented issues (R, F, A series)
├── pdf_extractor.py … chunk_builder.py     pipeline scripts
├── build_embeddings.py         FAISS index
├── retriever.py                hybrid retriever (v2.4.1, frozen)
├── eval_golden.py              evaluation + ablation
├── facts_db.py … financials_db.py   SQL track (facts, SEC XBRL, validation, serving layer)
├── answer.py                   answer layer (model chain, prompt, citations, cache)
├── eval_answers.py             answer evaluation with LLM judge
├── api.py                      FastAPI service
└── ask.py                      interactive questions
tests/                          offline unit and API tests (pytest)
.github/workflows/tests.yml     CI: runs the tests on every push
requirements.txt                pinned runtime dependencies
```

## Roadmap to 10 December 2026

| Dates | Work |
|-------|------|
| 12–18 Nov | SQL facts layer: question → company/year/item → value; numeric-question test |
| 19–26 Nov | Gemini answer layer with citations, "not stated in the filing" guard, response cache |
| 27 Nov | Automated tests, CI, HTTP API, pinned dependencies |
| 28 Nov–1 Dec | Final evaluation: prompt v4 on the dev set, 60 held-out questions with human labels |
| 2–5 Dec | Streamlit app: query → answer → source chunk + page |
| 6–7 Dec | Docker and deployment; updated limitations |
| 8–10 Dec | Report and presentation |

## Scope note

The approved proposal referenced a Kaggle sales database for the SQL track and LangChain for orchestration. During implementation these were replaced with a facts database built from the filings' own tables and a purpose-built retriever, so that the SQL track answers questions about the actual documents and every retrieval feature can be measured in isolation. Live market APIs, authentication and rate limiting are deferred to future work. These changes are being finalised with the supervisor.