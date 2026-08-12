import pickle
import re
from pathlib import Path

import faiss

from build_embeddings import EmbeddingIndexBuilder


class Retriever:

    # Words that signal the person wants the LATEST year specifically --
    # pure semantic similarity can't detect this (a 2016 chunk can be
    # just as "textually similar" to the query as a 2024 chunk), so we
    # handle it explicitly with a metadata-based re-rank.
    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|"
        r"newest|up[- ]to[- ]date)\b",
        re.IGNORECASE,
    )

        # Matches a standalone 4-digit year in the query (e.g. "in 2020").
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

    # Minimum cosine similarity below which we don't trust the top
    # match at all -- catches out-of-domain queries that still score
    # deceptively high (e.g. "Tesla revenue" scoring ~0.66).
    MIN_CONFIDENCE_SCORE = 0.58

    def __init__(self, store_dir="STAGE_1/vector_store"):

        store_dir = Path(store_dir)

        print("Loading FAISS index and metadata...")

        self.index = faiss.read_index(str(store_dir / "index.faiss"))

        with open(store_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.embedder = EmbeddingIndexBuilder()

        # The set of companies actually indexed -- used to detect when
        # a query mentions a company that ISN'T in the corpus at all
        # (e.g. asking about Tesla when only Apple is indexed), so we
        # can say "not available" instead of confidently returning the
        # wrong company's data.
        self.known_companies = sorted({
            chunk.get("company") for chunk in self.metadata
            if chunk.get("company")
        })

    # =========================================================
    # DETECT COMPANY MENTIONED IN THE QUERY
    # =========================================================

    def _detect_company(self, query):

        query_lower = query.lower()

        for company in self.known_companies:

            if company.lower() in query_lower:
                return company

        return None

    def _mentions_unknown_company(self, query):
        """
        Very lightweight check: if the query names a company that is
        clearly NOT any of the indexed companies, flag it. This can't
        catch every possible company name, but it catches the common
        case (a well-known company name appearing in the query that
        doesn't match anything in known_companies).
        """

        # A small, extendable list of well-known companies to check
        # against. This only matters for the "reject if truly foreign
        # company" case -- extend this list as more companies are
        # indexed in later stages.
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

        # -----------------------------------------------------
        # Company constraint: if the query clearly names a company
        # that isn't indexed at all, don't pretend to have an answer.
        # -----------------------------------------------------

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

        # Pull a larger candidate pool whenever we plan to filter/
        # re-rank afterwards, so filtering doesn't leave us with too
        # few results.
        needs_larger_pool = wants_recent or wanted_company or wanted_year
        pool_size = top_k * 6 if needs_larger_pool else top_k

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

        # -----------------------------------------------------
        # Company filter: hard filter, since a numeric answer from
        # the WRONG company is worse than no answer at all.
        # -----------------------------------------------------

        if wanted_company:

            filtered = [c for c in candidates if c["company"] == wanted_company]

            if filtered:
                candidates = filtered

        # -----------------------------------------------------
        # Year filter: hard filter when an explicit year is named --
        # this is different from the "recency" case (vague "latest"),
        # here the person named a SPECIFIC year, so only that year's
        # chunks should count.
        # -----------------------------------------------------

        if wanted_year:

            filtered = [c for c in candidates if c["year"] == wanted_year]

            if filtered:
                candidates = filtered

        if wants_recent and candidates:

            # Among the retrieved candidates, prefer the most recent
            # year, breaking ties by the original semantic score.
            # This doesn't touch the embeddings/index at all -- it's
            # a lightweight re-rank on top of normal vector search.
            max_year = max(
                (c["year"] for c in candidates if c["year"] is not None),
                default=None,
            )

            if max_year is not None:

                candidates.sort(
                    key=lambda c: (
                        c["year"] == max_year,
                        c["score"],
                    ),
                    reverse=True,
                )
        else:

            candidates.sort(key=lambda c: c["score"], reverse=True)

        results = candidates[:top_k]

        # Confidence gate -- if the best result still isn't good
        # enough, treat this as "no real answer" rather than
        # confidently returning weak matches.
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


# =============================================================
# QUICK MANUAL TEST
# =============================================================

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

                print(f"    \u26a0\ufe0f  {response['reason']}")

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

    # Save everything to a file too, so it's easy to review all
    # answers at once instead of scrolling through the terminal.
    import json as _json

    with open("retriever_test_results.json", "w", encoding="utf-8") as f:
        _json.dump(all_results_log, f, indent=2, ensure_ascii=False, default=str)

    print("\n\nFull results also saved to: retriever_test_results.json")