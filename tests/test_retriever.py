import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.retriever import hybrid_search

queries = [

    "What was Apple's total revenue in 2022?",
    "What risks did Apple mention?",
    "What was Apple's cash balance?",
    "What was Apple's operating cash flow?",
    "How much net income did Apple report in 2022?",
    "What were Apple's total assets?",
    "What were Apple's total liabilities?",
    "How much did Apple spend on research and development?",
    "What was iPhone revenue in 2022?",
    "What was Services revenue in 2022?",
    "What was Mac revenue in 2022?",
    "Did Apple discuss inflation risks?",
    "Did Apple mention supply chain risks?",
    "What legal proceedings did Apple disclose?",
    "What were Apple's deferred revenues?",
    "What share repurchase program did Apple announce?",
    "What were Apple's financing activities?",
    "What were Apple's investing activities?",
    "What products were launched in 2022?",
    "What did management discuss about financial performance?"

]

for query in queries:

    print("\n" + "=" * 100)
    print("QUERY:")
    print(query)
    print("=" * 100)

    results = hybrid_search(
        query=query,
        company="Apple",
        year=2022,
        top_k=5
    )

    if len(results) == 0:
        print("No Results Found.")
        continue

    for rank, result in enumerate(results, start=1):

        meta = result["metadata"]

        print(f"\nRank {rank}")
        print("-" * 80)
        print("Chunk ID :", meta["chunk_id"])
        print("Section  :", meta["section"])
        print("Year     :", meta["year"])
        print("Score    :", round(result["score"], 4))
        print("\nChunk Preview:\n")
        print(meta["text"][:500])