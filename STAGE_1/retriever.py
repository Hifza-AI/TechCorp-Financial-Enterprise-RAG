import json as _json
import pickle
import re
from pathlib import Path

import faiss

from build_embeddings import EmbeddingIndexBuilder

class Retriever:

    # Words that signal the person wants the LATEST year specifically :
    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|"
        r"newest|up[- ]to[- ]date)\b",
        re.IGNORECASE,
    )

    # FIX 1 (Requested): Non-capturing group used for exact 4-digit year match
    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    # Minimum cosine similarity threshold
    MIN_CONFIDENCE_SCORE = 0.58

    def __init__(self, store_dir="STAGE_1/vector_store"):

        store_dir = Path(store_dir)

        print("Loading FAISS index and metadata...")

        self.index = faiss.read_index(str(store_dir / "index.faiss"))

        with open(store_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.embedder = EmbeddingIndexBuilder()

        # Unique set of indexed companies
        self.known_companies = sorted({
            chunk.get("company") for chunk in self.metadata
            if chunk.get("company")
        })

        # Resolve recency against the FULL corpus metadata
        self.latest_year_by_company = {}
        for chunk in self.metadata:
            company = chunk.get("company")
            year = chunk.get("year")
            if company and year:
                current = self.latest_year_by_company.get(company, 0)
                self.latest_year_by_company[company] = max(current, year)

        self.global_latest_year = (
            max(self.latest_year_by_company.values())
            if self.latest_year_by_company else None
        )

    def _detect_company(self, query):

        query_lower = query.lower()

        for company in self.known_companies:
            if company.lower() in query_lower:
                return company

        return None

    def _mentions_unknown_company(self, query):

        WELL_KNOWN_COMPANIES = [
            "tesla", "amazon", "google", "microsoft", "meta", "nvidia",
            "netflix", "walmart", "jpmorgan", "citigroup", "pfizer",
        ]

        query_lower = query.lower()

        for name in WELL_KNOWN_COMPANIES:
            if name in query_lower:
                if not any(name in c.lower() for c in self.known_companies):
                    return name

        return None

    def search(self, query, top_k=5):

        # Company check for unindexed target entity
        unknown_company = self._mentions_unknown_company(query)

        if unknown_company:
            return {
                "matched": False,
                "reason": f"No indexed data for '{unknown_company}'.",
                "results": [],
            }

        wanted_company = self._detect_company(query)
        wants_recent = bool(self.RECENCY_KEYWORDS.search(query))

        year_match = self.YEAR_PATTERN.search(query)
        wanted_year = int(year_match.group()) if year_match else None

        # Anchor recency intent directly to full corpus actual latest year
        # FIX 2 (Requested): Explicit None check for wanted_year
        if wants_recent and wanted_year is None:
            if wanted_company and wanted_company in self.latest_year_by_company:
                wanted_year = self.latest_year_by_company[wanted_company]
            else:
                wanted_year = self.global_latest_year

        # FIX 2 (Requested): Explicit None check for wanted_year
        needs_hard_filter = bool(wanted_company or (wanted_year is not None))

        # Search whole index for metadata constraints to avoid recall truncation
        if needs_hard_filter:
            pool_size = self.index.ntotal
        else:
            pool_size = top_k

        query_vector = self.embedder.embed_query(query)

        scores, indices = self.index.search(query_vector, pool_size)

        candidates = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            chunk = self.metadata[idx]

            candidates.append({
                "score": float(score),
                "company": chunk.get("company"),
                "year": chunk.get("year"),
                "section_path": chunk.get("section_path"),
                "page_numbers": chunk.get("page_numbers"),
                "chunk_type": chunk.get("chunk_type"),
                "text": chunk.get("text"),
            })

        # Hard filter without silent fallbacks
        if wanted_company:
            candidates = [c for c in candidates if c["company"] == wanted_company]
            if not candidates:
                return {
                    "matched": False,
                    "reason": f"No data found for company '{wanted_company}'.",
                    "results": [],
                }

        # FIX 2 (Requested): Explicit None check for wanted_year
        if wanted_year is not None:
            candidates = [c for c in candidates if c["year"] == wanted_year]
            if not candidates:
                return {
                    "matched": False,
                    "reason": f"No data found for year {wanted_year}.",
                    "results": [],
                }

        candidates.sort(key=lambda c: c["score"], reverse=True)

        results = candidates[:top_k]

        if not results or results[0]["score"] < self.MIN_CONFIDENCE_SCORE:
            return {
                "matched": False,
                "reason": "No sufficiently relevant data found for this query.",
                "results": [],
            }

        return {
            "matched": True,
            "reason": None,
            "results": results,
        }


if __name__ == "__main__":

    retriever = Retriever()

    test_questions = {
        "Simple Numeric (single year)": [
            "What was Apple's total net sales in 2020?",
            "What was Apple's net income in 2019?",
            "What was Apple's gross margin in 2021?",
            "How much cash and cash equivalents did Apple have in 2022?",
        ],
        "Table-Based Lookup": [
            "What was Apple's total assets on the balance sheet?",
            "What was Apple's total liabilities?",
            "How much did Apple spend on selling, general and administrative expenses?",
            "What was Apple's earnings per share (EPS)?",
        ],
        "Trend / Comparison (multi-year)": [
            "How did Apple's iPhone sales change between 2019 and 2020?",
            "How has Apple's Services revenue grown over the years?",
            "Compare Apple's R&D spending in 2018 and 2021.",
        ],
        "Conceptual / Risk Factors": [
            "What risks does Apple face from supply chain disruptions?",
            "What litigation risks does Apple disclose?",
            "How does Apple describe competition risk in its business?",
            "What data privacy risks does Apple mention?",
        ],
        "Geographic / Segment": [
            "How much revenue did Apple generate in Greater China?",
            "What were Apple's net sales in Europe?",
        ],
        "Business Description": [
            "What products and services does Apple sell?",
            "What is Apple's business strategy?",
        ],
        "Recency-Aware": [
            "What is Apple's most recent total net sales?",
            "What is the latest R&D spending reported by Apple?",
        ],
        "Out-of-Scope (should NOT confidently match)": [
            "What is Apple's current stock price today?",
            "Who is the CEO of Microsoft?",
            "What is the weather in Cupertino?",
        ],
    }

    all_results_log = []

    for category, questions in test_questions.items():

        print("\n" + "#" * 70)
        print(f"# CATEGORY: {category}")
        print("#" * 70)

        for question in questions:

            print("\n" + "=" * 70)
            print("Q:", question)
            print("=" * 70)

            response = retriever.search(question, top_k=3)

            if not response["matched"]:

                print(f"    ⚠️  {response['reason']}")

                all_results_log.append({
                    "category": category,
                    "question": question,
                    "matched": False,
                    "reason": response["reason"],
                    "results": [],
                })

                continue

            results = response["results"]

            for i, r in enumerate(results, 1):

                print(
                    f"\n[{i}] score={r['score']:.3f} | "
                    f"{r['company']} {r['year']} | "
                    f"{r['chunk_type']} | {r['section_path']}"
                )

                preview = r["text"][:200].replace("\n", " ")

                print(f"    {preview}...")

            all_results_log.append({
                "category": category,
                "question": question,
                "matched": True,
                "reason": None,
                "results": results,
            })

    with open("retriever_test_results.json", "w", encoding="utf-8") as f:
        _json.dump(all_results_log, f, indent=4, ensure_ascii=False)
    
print("\n\nFull results also saved to: retriever_test_results.json")