"""
diagnose_table_boost.py

Follow-up to diagnose_boilerplate_bug.py. The b-parameter (BM25
length-normalization) experiment did NOT explain the failures --
in 2 of 3 cases it made things worse. The real, confirmed signal
from that test was different: for "What is Meta's Family of Apps
revenue?", the DENSE embedding ranks the correct TABLE chunk at
rank 35, while a generic prose paragraph mentioning the same term
ranks at dense_rank=0 -- a 35-rank gap that BM25 barely
differentiates (14 vs 15). This points to dense embeddings
under-representing table-heavy chunks for natural-language queries,
not a BM25 parameter issue.

This script tests ONE targeted, narrow fix: in the RRF fusion,
give 'table' chunk_type candidates a small fixed bonus ONLY when
the query looks like it's asking for a specific number (contains
words like "revenue", "income", "assets", "expense", etc. -- a
"give me a metric" query, not a conceptual/risk-factor query).
This is deliberately narrow so it does NOT touch the Risk-Factor
style queries (where table-boost would be irrelevant or harmful).

Does NOT modify retriever.py -- prints a side-by-side rank
comparison first, so we can see the real effect before deciding
whether to apply it for real. Full output saved to
diagnose_table_boost_results.txt.

USAGE:
    python diagnose_table_boost.py
"""

import re

from retriever import Retriever


OUTPUT_FILE = "diagnose_table_boost_results.txt"


METRIC_QUERY_PATTERN = re.compile(
    r"\b(revenue|net income|net sales|earnings|eps|assets|liabilities|"
    r"expense|profit|margin|cash|equity|income|sales)\b",
    re.IGNORECASE,
)

TABLE_BOOST = 0.03  # increased from 0.01 -- previous test showed
# correct direction but insufficient magnitude (Meta: rank 17->12,
# Intel: rank 91->51, neither reached top-3). Testing a stronger
# push specifically for these two more-winnable cases.


# (query, substring that must appear in the correct chunk)
TEST_CASES = [
    (
        "What is Meta's Family of Apps revenue?",
        "Family of Apps -- 2024: 162,355",
    ),
    (
        "What was Intel's net income in 2024?",
        "Net income (loss) -- Dec 28, 2024: (19,233)",
    ),
    # REGRESSION CHECK -- this query was ALREADY working correctly
    # (rank 0 before any boost). Confirms the boost doesn't push a
    # worse table over an already-correct table for the same query.
    (
        "What was Apple's total net sales in fiscal 2024?",
        "Total net sales -- 2024: 391,035",
    ),
    # Nvidia quarterly-revenue case intentionally dropped from this
    # round -- previous test showed a rank-2557 starting point, too
    # deep for any reasonable boost to fix safely. Likely a genuine
    # content-scope limitation (recent Nvidia filings may not
    # include a "Quarterly Summary" table at all, since SEC made
    # that disclosure optional after 2020-21), not a ranking bug --
    # revisit separately, don't force it via an oversized boost that
    # could corrupt unrelated queries.
]


def main():

    lines_out = []

    def log(msg=""):
        print(msg)
        lines_out.append(msg)

    retriever = Retriever()

    for query, correct_substring in TEST_CASES:

        log("\n" + "=" * 80)
        log(f"Query: {query}")
        log(f"is_metric_query: {bool(METRIC_QUERY_PATTERN.search(query))}")

        correct_idx = None
        for idx, chunk in enumerate(retriever.metadata):
            if correct_substring in chunk.get("text", ""):
                correct_idx = idx
                break

        if correct_idx is None:
            log("  !! correct chunk not found with this substring")
            continue

        dense_rank, dense_score = retriever._dense_search(query)
        bm25_rank = retriever._bm25_search(query)
        rrf_scores = retriever._reciprocal_rank_fusion(dense_rank, bm25_rank)

        ranked_before = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        rank_before = next(
            (i for i, (idx, _) in enumerate(ranked_before) if idx == correct_idx),
            None,
        )
        log(f"  BEFORE (no boost) -- correct chunk RRF rank: {rank_before}")
        log(f"  BEFORE -- current #1: chunk_type="
            f"{retriever.metadata[ranked_before[0][0]].get('chunk_type')}  "
            f"section={retriever.metadata[ranked_before[0][0]].get('section_path')}")

        is_metric_query = bool(METRIC_QUERY_PATTERN.search(query))

        boosted_scores = dict(rrf_scores)
        if is_metric_query:
            for idx in boosted_scores:
                if retriever.metadata[idx].get("chunk_type") == "table":
                    boosted_scores[idx] += TABLE_BOOST

        ranked_after = sorted(
            boosted_scores.items(), key=lambda x: x[1], reverse=True
        )
        rank_after = next(
            (i for i, (idx, _) in enumerate(ranked_after) if idx == correct_idx),
            None,
        )
        log(f"  AFTER (table-boost={'ON' if is_metric_query else 'OFF (not a metric query)'}) "
            f"-- correct chunk RRF rank: {rank_after}")
        log(f"  AFTER -- new #1: chunk_type="
            f"{retriever.metadata[ranked_after[0][0]].get('chunk_type')}  "
            f"section={retriever.metadata[ranked_after[0][0]].get('section_path')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"\n\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()