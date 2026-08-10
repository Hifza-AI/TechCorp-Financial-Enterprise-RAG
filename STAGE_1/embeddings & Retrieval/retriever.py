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

    def __init__(self, store_dir="STAGE_1/vector_store"):

        store_dir = Path(store_dir)

        print("Loading FAISS index and metadata...")

        self.index = faiss.read_index(str(store_dir / "index.faiss"))

        with open(store_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.embedder = EmbeddingIndexBuilder()

    def search(self, query, top_k=5):

        wants_recent = bool(self.RECENCY_KEYWORDS.search(query))

        # Pull a larger candidate pool when recency matters, so we
        # have enough options to re-rank by year without losing
        # semantic relevance.
        pool_size = top_k * 4 if wants_recent else top_k

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

        return candidates[:top_k]


# =============================================================
# QUICK MANUAL TEST
# =============================================================

if __name__ == "__main__":

    retriever = Retriever()

    test_questions = [
        "What were Apple's iPhone net sales in 2021?",
        "What risks does Apple face related to foreign currency?",
        "How much did Apple spend on research and development?",
        "What is Apple's total net sales for the most recent fiscal year?",
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print("Q:", question)
        print("=" * 70)

        results = retriever.search(question, top_k=3)

        for i, r in enumerate(results, 1):

            print(
                f"\n[{i}] score={r['score']:.3f} | "
                f"{r['company']} {r['year']} | "
                f"{r['chunk_type']} | {r['section_path']}"
            )

            preview = r["text"][:200].replace("\n", " ")

            print(f"    {preview}...")