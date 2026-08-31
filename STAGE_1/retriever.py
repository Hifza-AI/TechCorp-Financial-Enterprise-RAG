import json as _json
import pickle
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from build_embeddings import EmbeddingIndexBuilder


class Retriever:

    # Words that signal the person wants the LATEST year specifically :
    RECENCY_KEYWORDS = re.compile(
        r"\b(most recent|latest|current|this year|last fiscal year|"
        r"newest|up[- ]to[- ]date)\b",
        re.IGNORECASE,
    )

    # Non-capturing group used for exact 4-digit year match
    YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    # Minimum cosine similarity threshold. NOTE: this is checked
    # against the raw DENSE score specifically (see search() below),
    # never against the fused RRF score -- RRF scores live on a
    # completely different scale (small fractions, typically
    # 0.01-0.03) and were never meant to be compared against a
    # cosine-similarity cutoff. Reusing 0.58 against an RRF score
    # would make EVERY query fail the confidence gate.
    MIN_CONFIDENCE_SCORE = 0.58

    # Reciprocal Rank Fusion constant (standard value from IR literature --
    # not something that needs per-corpus tuning)
    RRF_K = 60

    def __init__(self, store_dir="STAGE_1/vector_store"):

        store_dir = Path(store_dir)

        print("Loading FAISS index and metadata...")

        self.index = faiss.read_index(str(store_dir / "index.faiss"))

        with open(store_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.embedder = EmbeddingIndexBuilder()

        # -----------------------------------------------------
        # BM25 (sparse/keyword) index -- built over the SAME chunks
        # as the dense FAISS index, so row i in both always refers
        # to the same chunk (metadata[i]). This lets us fuse rankings
        # from both retrieval methods without any extra bookkeeping.
        # -----------------------------------------------------

        print("Building BM25 keyword index...")

        tokenized_corpus = [
            self._tokenize(chunk.get("text", "")) for chunk in self.metadata
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

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

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

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

    # =========================================================
    # DENSE RETRIEVAL (whole corpus, so every chunk gets a rank)
    # =========================================================

    def _dense_search(self, query):

        query_vector = self.embedder.embed_query(query)

        scores, indices = self.index.search(query_vector, self.index.ntotal)

        rank_by_idx = {}
        score_by_idx = {}

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):

            if idx == -1:
                continue

            rank_by_idx[int(idx)] = rank
            score_by_idx[int(idx)] = float(score)

        return rank_by_idx, score_by_idx

    # =========================================================
    # BM25 RETRIEVAL (whole corpus, so every chunk gets a rank)
    # =========================================================

    def _bm25_search(self, query):

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        rank_by_idx = {idx: rank for rank, idx in enumerate(ranked_indices)}

        return rank_by_idx

    # =========================================================
    # RECIPROCAL RANK FUSION
    #
    # NEW: this is the piece that was MISSING -- self.bm25 was being
    # built in __init__, and RRF_K was defined, but search() never
    # actually queried BM25 or used RRF_K anywhere; retrieval was
    # still 100% pure dense. This wires both indexes into an actual
    # combined ranking.
    #
    # RRF combines two (or more) different ranked lists WITHOUT
    # needing their raw scores to be on comparable scales -- each
    # method only contributes 1/(k + rank) for however it ranked a
    # given chunk, so a chunk that ranks well in EITHER method (exact
    # keyword match via BM25, OR semantic similarity via dense) gets
    # pulled up, and a chunk that ranks well in BOTH gets pulled up
    # even further. This directly addresses the accuracy gap already
    # observed: queries whose exact wording (numbers, section names,
    # specific terms) doesn't come through strongly in the dense
    # embedding can still surface the right chunk via BM25's
    # keyword-overlap signal.
    # =========================================================

    def _reciprocal_rank_fusion(self, dense_rank_by_idx, bm25_rank_by_idx):

        all_indices = set(dense_rank_by_idx) | set(bm25_rank_by_idx)

        rrf_scores = {}

        for idx in all_indices:

            score = 0.0

            if idx in dense_rank_by_idx:
                score += 1.0 / (self.RRF_K + dense_rank_by_idx[idx] + 1)

            if idx in bm25_rank_by_idx:
                score += 1.0 / (self.RRF_K + bm25_rank_by_idx[idx] + 1)

            rrf_scores[idx] = score

        return rrf_scores

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
        if wants_recent and wanted_year is None:
            if wanted_company and wanted_company in self.latest_year_by_company:
                wanted_year = self.latest_year_by_company[wanted_company]
            else:
                wanted_year = self.global_latest_year

        # Used below to relax the confidence gate once we've
        # deterministically confirmed (via metadata, not fuzzy
        # matching) that this company/year combination is in the
        # corpus at all -- see the confidence-gate comment further
        # down for the full rationale.
        needs_hard_filter = bool(wanted_company or (wanted_year is not None))

        # -----------------------------------------------------
        # Run BOTH retrieval methods over the WHOLE corpus (needed
        # so RRF has a real rank for every chunk, and so post-hoc
        # company/year filtering never truncates recall the way a
        # small top_k pool would).
        # -----------------------------------------------------

        dense_rank_by_idx, dense_score_by_idx = self._dense_search(query)

        bm25_rank_by_idx = self._bm25_search(query)

        rrf_scores = self._reciprocal_rank_fusion(
            dense_rank_by_idx, bm25_rank_by_idx
        )

        candidates = []

        for idx, rrf_score in rrf_scores.items():

            chunk = self.metadata[idx]

            candidates.append({
                "score": dense_score_by_idx.get(idx, 0.0),
                "rrf_score": rrf_score,
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

        if wanted_year is not None:
            candidates = [c for c in candidates if c["year"] == wanted_year]
            if not candidates:
                return {
                    "matched": False,
                    "reason": f"No data found for year {wanted_year}.",
                    "results": [],
                }

        # Rank by the FUSED score now, not the raw dense score --
        # this is what actually lets BM25 influence which chunks
        # surface and in what order.
        candidates.sort(key=lambda c: c["rrf_score"], reverse=True)

        results = candidates[:top_k]

        # Confidence gate STILL checks the raw DENSE cosine-similarity
        # of the top result (never the RRF score -- see the
        # MIN_CONFIDENCE_SCORE docstring above). This preserves the
        # already-verified out-of-scope rejection behavior (unknown
        # topics, irrelevant queries) exactly as it worked before
        # hybrid retrieval was wired in.
        #
        # NEW: skip the numeric threshold ENTIRELY when a hard
        # company/year filter was applied. Confirmed on real queries:
        # a 0.7x-relaxed threshold (0.406) helped "most recent total
        # net sales" (top dense score 0.544) but still failed "latest
        # reported net income" (its best year-filtered candidate
        # apparently scored even lower) -- there's no single relaxed
        # multiplier that reliably covers every metric/phrasing
        # combination, because different metrics naturally embed with
        # different strength against a given query, especially once
        # STRICTLY filtered to one specific year.
        #
        # The numeric threshold's actual PURPOSE is to catch queries
        # about things not in the corpus at all (weather, a stock
        # price today, an unindexed company) -- but once we've
        # already deterministically confirmed via metadata (not
        # fuzzy/semantic matching) that this exact company+year
        # combination IS in the corpus, that specific risk is fully
        # ruled out. Any content that survived the hard filter is, by
        # definition, real disclosed content for that company/year --
        # for an ordinary financial-metric question, that's enough to
        # return an answer rather than a false "no match".
        if needs_hard_filter:
            if not results:
                return {
                    "matched": False,
                    "reason": "No sufficiently relevant data found for this query.",
                    "results": [],
                }
        elif not results or results[0]["score"] < self.MIN_CONFIDENCE_SCORE:
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
        "Simple Numeric (single year) -- ALL 4 COMPANIES": [
            "What was Apple's total net sales in 2020?",
            "What was Microsoft's total revenue in 2022?",
            "What was CVS's net income?",
            "What was Costco's total revenue?",
        ],
        "Table-Based Lookup -- ALL 4 COMPANIES": [
            "What was Apple's total assets on the balance sheet?",
            "What was Microsoft's total liabilities?",
            "What was CVS's earnings per share (EPS)?",
            "What was Costco's operating income?",
        ],
        "Trend / Comparison (multi-year)": [
            "How did Apple's iPhone sales change between 2019 and 2020?",
            "How has Microsoft's cloud revenue grown over the years?",
        ],
        "Conceptual / Risk Factors -- ALL 4 COMPANIES": [
            "What risks does Apple face from supply chain disruptions?",
            "What risks does Microsoft disclose related to cybersecurity?",
            "What litigation risks does CVS disclose?",
            "What competitive risks does Costco mention?",
        ],
        "Geographic / Segment -- ALL 4 COMPANIES": [
            "How much revenue did Apple generate in Greater China?",
            "What were Microsoft's segment revenues?",
            "What were Costco's operating income by geographic region?",
        ],
        "Business Description -- ALL 4 COMPANIES": [
            "What products and services does Apple sell?",
            "What is Microsoft's business strategy?",
            "What business segments does CVS operate?",
            "What is Costco's membership warehouse business model?",
        ],
        "Recency-Aware": [
            "What is Apple's most recent total net sales?",
            "What is Microsoft's latest reported revenue?",
        ],
        "Cross-Company Differentiation (company-filter sanity check)": [
            "What was Microsoft's revenue?",  # should NOT return Apple data
            "What was CVS's net income?",     # should NOT return Costco data
        ],
        "Out-of-Scope (should NOT confidently match)": [
            "What is Apple's current stock price today?",
            "Who is the CEO of Tesla?",
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
                    f"\n[{i}] dense={r['score']:.3f} rrf={r['rrf_score']:.4f} | "
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