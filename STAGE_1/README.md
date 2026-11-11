# TechCorp Financial Enterprise RAG

A local, verifiable Retrieval-Augmented Generation (RAG) knowledge base over **500 SEC 10-K annual reports** (50 US public companies × 10 fiscal years). Every answer is traceable to a company, fiscal year, section and page number in the original filing.

**Final Year Project** — BS Computer Science, Emerson University Multan
**Supervisor:** Engr. Dr. Muhammad Naveed Yasir
**Team:** Hifza Riaz (COSC232112112), Saman Saif (COSC232112102)

---

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
| 9 | Hybrid retriever + evaluation suite | ✅ Complete (v2.3) | `retriever.py`, `eval_golden.py`, `ask.py` |
| 10 | SQL facts layer (exact numeric lookup) | 🔄 In progress | `facts_db.py` |
| 11 | Answer layer: Gemini + citations + "not stated" guard | ⬜ Planned | — |
| 12 | Streamlit app | ⬜ Planned | — |
| 13 | Deployment (Docker) | ⬜ Optional | — |

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

> The v2.3 fixes were designed from this set's misses, so 90/100 is an in-sample figure. The unbiased number is 80/100. A fresh held-out set is planned for the final evaluation.

Known issues are documented in [`STAGE_1/known_limitations.md`](STAGE_1/known_limitations.md) (R1–R20).

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
```

To rebuild from PDFs, run the pipeline scripts in the order listed in the table above (`pdf_extractor.py` → … → `build_embeddings.py`). Each script reads the previous stage's folder under `STAGE_1/` and writes its own.

### Requirements

Python 3.12 · `pymupdf` · `sentence-transformers` · `faiss-cpu` · `rank-bm25` · `numpy`

```bash
pip install pymupdf sentence-transformers faiss-cpu rank-bm25 numpy
```

## Repository layout

```
STAGE_1/
├── data/pdfs/<Company>/        original 10-K PDFs
├── extracted/ cleaned/ heading_detection/ table_analysis/
├── parsed_tables/ paragraphs/ paragraphs_fixed/ hierarchy/
├── chunks/<Company>/           final chunks (JSON)
├── eval/                       test results and ablation outputs
├── test_questions.csv          frozen 100-question fair test
├── known_limitations.md        documented issues R1–R20
├── pdf_extractor.py … chunk_builder.py     pipeline scripts
├── build_embeddings.py         FAISS index
├── retriever.py                hybrid retriever (v2.3)
├── eval_golden.py              evaluation + ablation
├── facts_db.py                 SQL facts layer
└── ask.py                      interactive questions
```

## Roadmap to 10 December 2026

| Dates | Work |
|-------|------|
| 12–18 Nov | SQL facts layer: question → company/year/item → value; numeric-question test |
| 19–26 Nov | Gemini answer layer with citations, "not stated in the filing" guard, response cache |
| 27 Nov–3 Dec | Streamlit app: query → answer → source chunk + page |
| 4–7 Dec | Final evaluation on a fresh held-out set; updated limitations |
| 8–10 Dec | Report and presentation |

## Scope note

The approved proposal referenced a Kaggle sales database for the SQL track and LangChain for orchestration. During implementation these were replaced with a facts database built from the filings' own tables and a purpose-built retriever, so that the SQL track answers questions about the actual documents and every retrieval feature can be measured in isolation. Live market APIs, authentication and rate limiting are deferred to future work. These changes are being finalised with the supervisor.