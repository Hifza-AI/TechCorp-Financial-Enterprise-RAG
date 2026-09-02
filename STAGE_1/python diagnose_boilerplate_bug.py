"""
diagnose_boilerplate_bug.py

STEP 1 (diagnosis): For a few known-failing queries, prints the RAW
dense score AND raw BM25 score for BOTH (a) the chunk the retriever
actually returned at #1, and (b) the chunk we KNOW is the correct
answer (found via verify_chunks_exist.py). This tells us, in exact
numbers, whether BM25 or dense scoring is the one inflating the
boilerplate result -- no more guessing.

STEP 2 (candidate fix test): Re-runs the SAME queries using a
LOWERED BM25 length-normalization parameter (b=0.3 instead of the
rank_bm25 default of 0.75). Lower b means BM25 penalizes longer
chunks LESS -- the hypothesis is that short, keyword-dense
boilerplate paragraphs are winning purely because they're short, not
because they're more relevant. If this hypothesis is right, the
correct (often longer/table) chunk should score competitively once
length is de-emphasized.

This does NOT modify retriever.py yet -- it's a side-by-side
comparison so we can see the actual effect on real data BEFORE
changing the real file. Full output saved to
diagnose_boilerplate_results.txt.

USAGE:
    python diagnose_boilerplate_bug.py
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from retriever import Retriever


OUTPUT_FILE = "diagnose_boilerplate_results.txt"


# (query, expected-substring that should appear in the CORRECT chunk)
DIAGNOSTIC_CASES = [
    (
        "What is Meta's Family of Apps revenue?",
        "Family of Apps -- 2024: 162,355",  # the real Note 2 Revenue row
    ),
    (
        "What competitive risks does Meta mention?",
        "adversely affect our business",  # a real Risk Factors sentence, not boilerplate
    ),
    (
        "What risks does Intel disclose related to manufacturing?",
        "vulnerable to product and manufacturing-related risks",
    ),
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def main():

    lines_out = []

    def log(msg=""):
        print(msg)
        lines_out.append(msg)

    retriever = Retriever()

    log("=" * 80)
    log("STEP 1 -- DIAGNOSIS (current retriever, default BM25 b=0.75)")
    log("=" * 80)

    for query, correct_substring in DIAGNOSTIC_CASES:

        log(f"\nQuery: {query}")
        log(f"Looking for chunk containing: {correct_substring!r}")

        # Find the correct chunk's index directly in metadata
        correct_idx = None
        for idx, chunk in enumerate(retriever.metadata):
            if correct_substring in chunk.get("text", ""):
                correct_idx = idx
                break

        if correct_idx is None:
            log("  !! Could not locate the correct chunk in metadata "
                "-- adjust correct_substring.")
            continue

        # Run dense + BM25 the same way retriever.search() does
        dense_rank, dense_score = retriever._dense_search(query)
        bm25_rank = retriever._bm25_search(query)

        response = retriever.search(query, top_k=1)
        top_idx = None
        if response["matched"]:
            top_text = response["results"][0]["text"]
            for idx, chunk in enumerate(retriever.metadata):
                if chunk.get("text") == top_text:
                    top_idx = idx
                    break

        log(f"  Retriever's #1 result -- chunk_type="
            f"{retriever.metadata[top_idx].get('chunk_type') if top_idx is not None else '?'}"
            f"  dense_score={dense_score.get(top_idx, 0):.3f}"
            f"  dense_rank={dense_rank.get(top_idx, '-')}"
            f"  bm25_rank={bm25_rank.get(top_idx, '-')}")

        log(f"  Correct chunk (idx {correct_idx}) -- chunk_type="
            f"{retriever.metadata[correct_idx].get('chunk_type')}"
            f"  dense_score={dense_score.get(correct_idx, 0):.3f}"
            f"  dense_rank={dense_rank.get(correct_idx, '-')}"
            f"  bm25_rank={bm25_rank.get(correct_idx, '-')}")

        text_len_top = len(retriever.metadata[top_idx].get("text", "")) if top_idx is not None else 0
        text_len_correct = len(retriever.metadata[correct_idx].get("text", ""))
        log(f"  Text length -- winner: {text_len_top} chars | "
            f"correct: {text_len_correct} chars")

    log("\n" + "=" * 80)
    log("STEP 2 -- CANDIDATE FIX (BM25 b=0.3, less length-penalty)")
    log("=" * 80)

    tokenized_corpus = [tokenize(c.get("text", "")) for c in retriever.metadata]
    bm25_lowb = BM25Okapi(tokenized_corpus, b=0.3)

    for query, correct_substring in DIAGNOSTIC_CASES:

        log(f"\nQuery: {query}")

        correct_idx = None
        for idx, chunk in enumerate(retriever.metadata):
            if correct_substring in chunk.get("text", ""):
                correct_idx = idx
                break

        if correct_idx is None:
            continue

        tokenized_query = tokenize(query)
        scores = bm25_lowb.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        new_bm25_rank = {idx: rank for rank, idx in enumerate(ranked_indices)}

        log(f"  Correct chunk NEW bm25_rank (b=0.3): "
            f"{new_bm25_rank.get(correct_idx, '-')}  "
            f"(was {retriever._bm25_search(query).get(correct_idx, '-')} "
            f"with default b=0.75)")

        top_new_idx = ranked_indices[0]
        log(f"  New #1 by BM25 alone -- chunk_type="
            f"{retriever.metadata[top_new_idx].get('chunk_type')}  "
            f"section={retriever.metadata[top_new_idx].get('section_path')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"\n\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()