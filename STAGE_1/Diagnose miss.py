"""
diagnose_miss.py

For a query that didn't retrieve well, this answers the ONE question
that matters: is the correct chunk MISSING from the corpus entirely
(a chunking/extraction problem -- nothing the retriever can fix), or
does it EXIST but rank poorly (a retriever/embedding problem)?

HOW TO USE:
    Fill in a case in MISS_CASES below -- the query that failed, plus
    a short keyword/phrase you KNOW should appear in the chunk that
    actually has the right answer (e.g. a specific number, or a
    distinctive phrase from the real 10-K). Run the script.

WHAT IT DOES, for each case:
    1. Runs the query through BOTH dense search and BM25 search
       against the FULL corpus (no company/year filtering), so you
       see the true, unfiltered ranking.
    2. Searches metadata for any chunk containing your keyword,
       scoped to the right company (and year, if you gave one).
    3. Reports:
         - Does a matching chunk exist at all? (if NOT -> confirmed
           data/chunking issue, not a retriever issue)
         - If it exists: what dense rank, BM25 rank, and RRF score
           did it get, versus the rank of whatever the retriever
           actually returned instead
    4. Prints a plain verdict: "DATA ISSUE" or "RETRIEVER ISSUE" (or
       "PARTIAL -- ranks but not high enough") for each case, so you
       don't have to interpret the raw numbers yourself.

USAGE:
    python diagnose_miss.py
"""

from retriever import Retriever


# =====================================================
# Fill in cases here. `keyword` should be a short, DISTINCTIVE
# phrase or number you know appears in the chunk that has the real
# answer -- e.g. an exact dollar figure, or a section title fragment.
# `company` / `year` narrow the search to the right company/year so
# you don't accidentally match some OTHER company's similar text.
# =====================================================

MISS_CASES = [
    {
        "query": "What was Intel's total assets on the balance sheet?",
        "keyword": "Total assets",
        "company": "Intel",
        "year": None,  # None = don't filter by year, just check it exists ANYWHERE for this company
    },
    {
        "query": "What was ServiceNow's total assets on the balance sheet?",
        "keyword": "Total assets",
        "company": "ServiceNow",
        "year": None,
    },
    {
        "query": "What was Palo Alto Networks' total stockholders' equity?",
        "keyword": "Total stockholders",
        "company": "Palo Alto Networks",
        "year": None,
    },
    {
        "query": "What is Meta's lease liability maturity schedule?",
        "keyword": "lease liabilit",  # lowercase, partial word to catch "liability"/"liabilities"
        "company": "META",
        "year": None,
    },
    {
        "query": "What is Nvidia's inventory valuation policy?",
        "keyword": "inventor",  # catches "inventory"/"inventories"
        "company": "Nvidia",
        "year": None,
    },
    {
        "query": "What is Workday's deferred commissions accounting policy?",
        "keyword": "deferred commission",
        "company": "Workday",
        "year": None,
    },
    {
        "query": "What is ServiceNow's deferred revenue balance?",
        "keyword": "deferred revenue",
        "company": "ServiceNow",
        "year": None,
    },
    {
        "query": "What was Amazon's most recent net income?",
        "keyword": "Net income",
        "company": "Amazon",
        "year": 2025,
    },
    {
        "query": "What was Walmart's most recent total revenue?",
        "keyword": "Total revenue",
        "company": "Walmart",
        "year": 2026,
    },
    # Add more cases here as you find them -- same shape.
]


def find_matching_chunks(retriever, keyword, company=None, year=None):
    """
    Scans the FULL metadata for any chunk whose text contains
    `keyword` (case-insensitive), optionally narrowed to a specific
    company/year. Returns a list of (index, chunk) tuples.
    """

    keyword_lower = keyword.lower()
    matches = []

    for idx, chunk in enumerate(retriever.metadata):

        if company and chunk.get("company") != company:
            continue

        if year is not None and chunk.get("year") != year:
            continue

        text = (chunk.get("text") or "").lower()

        if keyword_lower in text:
            matches.append((idx, chunk))

    return matches


def diagnose_case(retriever, case):

    query = case["query"]
    keyword = case["keyword"]
    company = case.get("company")
    year = case.get("year")

    print("\n" + "=" * 80)
    print(f"CASE: {query}")
    print(f"Looking for chunks containing: {keyword!r}"
          + (f"  (company={company})" if company else "")
          + (f"  (year={year})" if year is not None else ""))
    print("=" * 80)

    # Step 1: does a matching chunk exist at all?
    matches = find_matching_chunks(retriever, keyword, company, year)

    if not matches:
        print(f"\n  RESULT: NO chunk containing {keyword!r} found for "
              f"{company or 'any company'}"
              + (f" / year {year}" if year is not None else "") + ".")
        print("  VERDICT: *** DATA / CHUNKING ISSUE ***")
        print("  The information genuinely isn't in the indexed corpus --")
        print("  this is NOT something the retriever can fix. Check the")
        print("  original chunks.json / table_analyzed.json for this")
        print("  company to see whether the value was ever extracted.")
        return

    print(f"\n  Found {len(matches)} chunk(s) containing {keyword!r}.")

    # Step 2: run the query unfiltered, get full dense + BM25 rankings
    dense_rank_by_idx, dense_score_by_idx = retriever._dense_search(query)
    bm25_rank_by_idx = retriever._bm25_search(query)
    rrf_scores = retriever._reciprocal_rank_fusion(
        dense_rank_by_idx, bm25_rank_by_idx
    )

    # Step 3: for each matching chunk, report where it ranked
    best_rrf_rank = None

    all_rrf_sorted = sorted(
        rrf_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    rrf_rank_by_idx = {idx: rank for rank, (idx, _) in enumerate(all_rrf_sorted)}

    for idx, chunk in matches:

        dense_rank = dense_rank_by_idx.get(idx, "not in top results")
        bm25_rank = bm25_rank_by_idx.get(idx, "not in top results")
        rrf_rank = rrf_rank_by_idx.get(idx, "not scored")

        if isinstance(rrf_rank, int) and (
            best_rrf_rank is None or rrf_rank < best_rrf_rank
        ):
            best_rrf_rank = rrf_rank

        preview = (chunk.get("text") or "")[:120].replace("\n", " ")

        print(f"\n  Chunk (row {idx}) -- section: {chunk.get('section_path')}")
        print(f"    dense_rank={dense_rank}  bm25_rank={bm25_rank}  "
              f"final_rrf_rank={rrf_rank}")
        print(f"    preview: {preview}...")

    # Step 4: verdict
    print()

    if best_rrf_rank is None:
        print("  VERDICT: *** RETRIEVER ISSUE (severe) ***")
        print("  The correct chunk exists but never scored on EITHER")
        print("  dense or BM25 ranking -- something about the query")
        print("  wording doesn't match this chunk at all semantically")
        print("  or lexically. Consider rephrasing the test query, or")
        print("  this may indicate an embedding-quality issue for this")
        print("  specific chunk (e.g. it may itself be poorly formed --")
        print("  check the chunk text above for garbling).")
    elif best_rrf_rank <= 2:
        print(f"  VERDICT: RANKS WELL (position {best_rrf_rank}) -- if this")
        print("  didn't show up in your top-3 result, double check company/")
        print("  year filtering logic, or this may already be fixed.")
    elif best_rrf_rank <= 10:
        print(f"  VERDICT: *** RETRIEVER ISSUE (moderate) ***")
        print(f"  Correct chunk exists and ranks at position {best_rrf_rank}")
        print("  overall -- close, but not making it into top_k=3. This is")
        print("  a genuine ranking/scoring problem, not a data problem.")
    else:
        print(f"  VERDICT: *** RETRIEVER ISSUE (severe) ***")
        print(f"  Correct chunk exists but ranks far down (position "
              f"{best_rrf_rank}) -- something is actively pulling the")
        print("  WRONG chunks above it. Check what's outranking it (the")
        print("  chunks actually returned in your test run) for shared")
        print("  keywords that might be over-weighted by BM25, or generic")
        print("  boilerplate phrasing that embeds deceptively close to")
        print("  the query.")


def main():

    print("Loading retriever (this builds the BM25 index, may take a moment)...")
    retriever = Retriever()

    for case in MISS_CASES:
        diagnose_case(retriever, case)

    print("\n\nDone. Review each case's VERDICT above.")


if __name__ == "__main__":
    main()