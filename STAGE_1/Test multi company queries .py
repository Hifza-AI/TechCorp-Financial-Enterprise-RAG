"""
test_multi_company_queries.py

Runs a focused set of test queries -- across ALL 5 indexed companies
(Apple, Google, Intel, Meta, Nvidia) -- against the Retriever, and
prints the TOP result for each so you can quickly eyeball whether
it's correct (right numbers, right section) against the real PDF.

Mirrors the exact group structure (A-G) used in the earlier
Apple-only test_apple_queries.py, extended to cover every company so
Recall@3 / Precision@1 can be measured the same way for direct
comparison against the original Apple-only baseline.

NEW: ALL output is written to test_multi_company_results.txt as well
as printed to the terminal -- long runs were getting cut off in the
terminal window, so the file is now the reliable, complete copy to
share for review. Chunk text is also no longer truncated to 250
chars -- the full text is shown, so it's possible to tell whether an
answer is genuinely absent vs just further down in a longer chunk.

USAGE:
    python test_multi_company_queries.py
"""

from retriever import Retriever


OUTPUT_FILE = "test_multi_company_results.txt"


TEST_QUERIES = [
    # =====================================================
    # Group A -- exact-number queries (BM25 should help most here)
    # =====================================================
    "What was Apple's total net sales in fiscal 2024?",
    "What was Apple's net income in 2024?",
    "What was Apple's diluted earnings per share in 2024?",
    "What was Google's total revenue in 2024?",
    "What was Google's net income in 2024?",
    "What was Intel's total revenue in 2024?",
    "What was Intel's net income in 2024?",
    "What was Meta's total revenue in 2024?",
    "What was Meta's diluted earnings per share in 2024?",
    "What was Nvidia's total revenue in fiscal 2025?",
    "What was Nvidia's net income in fiscal 2025?",

    # =====================================================
    # Group B -- table-based lookups
    # =====================================================
    "What was Apple's total assets on the balance sheet?",
    "What was Apple's revenue from iPhone in 2024?",
    "What was Google's total liabilities on the balance sheet?",
    "What was Google's revenue from Google Cloud?",
    "What was Intel's total assets on the balance sheet?",
    "What was Intel's research and development expense?",
    "What was Meta's total costs and expenses?",
    "What was Meta's cash and cash equivalents?",
    "What was Nvidia's gross profit?",
    "What was Nvidia's cash and cash equivalents?",

    # =====================================================
    # Group C -- geographic / segment
    # =====================================================
    "How much revenue did Apple generate in Greater China?",
    "What were Google's revenues by geography?",
    "What were Intel's operating segments?",
    "What is Meta's Family of Apps revenue?",
    "What was Nvidia's Data Center segment revenue?",

    # =====================================================
    # Group D -- Notes-specific
    # =====================================================
    "What is Apple's revenue recognition policy?",
    "What are Apple's deferred tax assets?",
    "What is Google's stock-based compensation expense?",
    "What is Intel's goodwill balance?",
    "What is Meta's lease liability maturity schedule?",
    "What is Nvidia's inventory valuation policy?",

    # =====================================================
    # Group E -- Risk Factors
    # =====================================================
    "What risks does Apple face related to product introductions and transitions?",
    "What risks does Google disclose related to competition?",
    "What risks does Intel disclose related to manufacturing?",
    "What competitive risks does Meta mention?",
    "What supply chain risks does Nvidia disclose?",

    # =====================================================
    # Group F -- recency-aware
    # =====================================================
    "What is Apple's most recent total net sales?",
    "What is Google's latest reported revenue?",
    "What is Nvidia's most recent quarterly revenue?",

    # =====================================================
    # Group G -- cross-company differentiation (company-filter sanity check)
    # =====================================================
    "What was Google's revenue?",    # should NOT return Apple data
    "What was Intel's net income?",  # should NOT return Meta data

    # =====================================================
    # Group H -- out-of-scope (should NOT confidently match)
    # =====================================================
    "What is Apple's current stock price today?",
    "Who is the CEO of Tesla?",
]


def main():

    retriever = Retriever()

    lines_out = []

    def log(msg=""):
        print(msg)
        lines_out.append(msg)

    for i, query in enumerate(TEST_QUERIES, 1):

        log("\n" + "=" * 80)
        log(f"[{i}] Q: {query}")
        log("=" * 80)

        response = retriever.search(query, top_k=3)

        if not response["matched"]:
            log(f"    NO MATCH -- {response['reason']}")
            continue

        for rank, r in enumerate(response["results"], 1):

            log(
                f"\n  #{rank}  dense={r['score']:.3f}  rrf={r['rrf_score']:.4f}  "
                f"| {r['company']} {r['year']} | {r['chunk_type']}"
            )
            log(f"      section: {r['section_path']}")
            log(f"      pages  : {r['page_numbers']}")

            # Full text, not truncated to 250 chars -- the earlier
            # truncated preview made it impossible to tell whether
            # the ACTUAL number/answer was further down in the
            # chunk's text or genuinely absent.
            log(f"      text   : {r['text']}")

        # Flush to disk after EVERY query, not just at the end -- if
        # the run gets interrupted partway through, whatever ran so
        # far is still saved and reviewable.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))

    print(f"\n\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()