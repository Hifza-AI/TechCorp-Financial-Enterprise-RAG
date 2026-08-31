"""
test_apple_queries.py

Runs a focused set of Apple-specific test queries against the
Retriever, and prints the TOP result for each so you can quickly
eyeball whether it's correct (right numbers, right section) against
the real PDF.

USAGE:
    python test_apple_queries.py
"""

from retriever import Retriever


APPLE_TEST_QUERIES = [
    # Group A -- exact-number queries (BM25 should help most here)
    "What was Apple's total net sales in fiscal 2024?",
    "What was Apple's net income in 2024?",
    "What was Apple's basic earnings per share in 2024?",
    "What was Apple's diluted earnings per share in 2024?",
    "How many shares did Apple repurchase in the third quarter of fiscal 2018?",

    # Group B -- table-based lookups
    "What was Apple's total assets on the balance sheet?",
    "What was Apple's revenue from iPhone in 2024?",
    "What was Apple's revenue from Services in 2024?",
    "What was Apple's gross property, plant and equipment in 2024?",
    "What was Apple's provision for income taxes in 2024?",

    # Group C -- geographic / segment
    "How much revenue did Apple generate in the Americas segment?",
    "How much revenue did Apple generate in Greater China?",
    "What was Apple's operating income in Europe?",

    # Group D -- Notes-specific
    "What is Apple's revenue recognition policy?",
    "What are Apple's deferred tax assets?",
    "What is Apple's lease liability maturity schedule?",
    "What did Apple disclose about the European Commission State Aid Decision?",

    # Group E -- Risk Factors
    "What risks does Apple face related to product introductions and transitions?",
    "How does Apple's business depend on distributors and resellers?",
    "What foreign exchange rate risks does Apple disclose?",

    # Group F -- recency-aware
    "What is Apple's most recent total net sales?",
    "What is Apple's latest reported net income?",

    # Group G -- out-of-scope (should NOT confidently match)
    "What is Apple's current stock price today?",
    "Who is the CEO of Tesla?",
]


def main():

    retriever = Retriever()

    for i, query in enumerate(APPLE_TEST_QUERIES, 1):

        print("\n" + "=" * 80)
        print(f"[{i}] Q: {query}")
        print("=" * 80)

        response = retriever.search(query, top_k=3)

        if not response["matched"]:
            print(f"    NO MATCH -- {response['reason']}")
            continue

        for rank, r in enumerate(response["results"], 1):

            print(
                f"\n  #{rank}  dense={r['score']:.3f}  rrf={r['rrf_score']:.4f}  "
                f"| {r['company']} {r['year']} | {r['chunk_type']}"
            )
            print(f"      section: {r['section_path']}")
            print(f"      pages  : {r['page_numbers']}")

            preview = r["text"][:250].replace("\n", " ")
            print(f"      text   : {preview}...")


if __name__ == "__main__":
    main()