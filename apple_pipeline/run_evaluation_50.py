import os
import json
import time
from typing import List, Dict, Any
# Import existing retriever class directly without modifying it
from retriever_test import EnterpriseRetriever, INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME

# ============================================
# 50 ENTERPRISE TEST CASES (GOLDEN BENCHMARK)
# ============================================

TEST_DATASET = [
    # --- Core Statements (Balance Sheets / Income / Cash Flow) ---
    {"id": 1, "query": "What was the net income in 2021?", "target_section": ["consolidated statements of comprehensive income", "statements of operations"], "target_year": "2021"},
    {"id": 2, "query": "What were total current assets in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},
    {"id": 3, "query": "What was operating cash flow in 2023?", "target_section": ["consolidated statements of cash flows"], "target_year": "2023"},
    {"id": 4, "query": "What was total gross margin in 2022?", "target_section": ["consolidated statements of operations", "gross margin"], "target_year": "2022"},
    {"id": 5, "query": "What was total current liabilities in 2021?", "target_section": ["consolidated balance sheets"], "target_year": "2021"},
    {"id": 6, "query": "What was net income for Apple in 2023?", "target_section": ["consolidated statements of operations", "comprehensive income"], "target_year": "2023"},
    {"id": 7, "query": "What were cash and cash equivalents in 2022?", "target_section": ["consolidated balance sheets", "cash flows"], "target_year": "2022"},
    {"id": 8, "query": "What was total operating income in 2021?", "target_section": ["statements of operations"], "target_year": "2021"},
    {"id": 9, "query": "What were long term debt amounts in 2022?", "target_section": ["balance sheets", "term debt"], "target_year": "2022"},
    {"id": 10, "query": "What were total assets in 2021?", "target_section": ["consolidated balance sheets"], "target_year": "2021"},

    # --- Operating Expenses & Specific Costs ---
    {"id": 11, "query": "What were research and development expenses in 2024?", "target_section": ["operating expenses", "research and development"], "target_year": "2024"},
    {"id": 12, "query": "What were selling general and administrative expenses in 2023?", "target_section": ["operating expenses"], "target_year": "2023"},
    {"id": 13, "query": "What was provision for income taxes in 2021?", "target_section": ["note 5 – income taxes", "income taxes"], "target_year": "2021"},
    {"id": 14, "query": "How much was spent on research and development in 2022?", "target_section": ["operating expenses", "research and development"], "target_year": "2022"},
    {"id": 15, "query": "What was the effective tax rate in 2021?", "target_section": ["note 5 – income taxes"], "target_year": "2021"},
    {"id": 16, "query": "What were depreciation and amortization costs in 2023?", "target_section": ["statements of cash flows"], "target_year": "2023"},
    {"id": 17, "query": "What were share based compensation expenses in 2023?", "target_section": ["statements of cash flows"], "target_year": "2023"},
    {"id": 18, "query": "What were research and development expenses in 2021?", "target_section": ["operating expenses"], "target_year": "2021"},

    # --- Geographic & Product Segments ---
    {"id": 19, "query": "What were net sales in Greater China in 2022?", "target_section": ["greater china", "segment information"], "target_year": "2022"},
    {"id": 20, "query": "What was total revenue for Services in 2023?", "target_section": ["services", "segment information"], "target_year": "2023"},
    {"id": 21, "query": "What were net sales in Europe segment in 2022?", "target_section": ["europe", "segment information"], "target_year": "2022"},
    {"id": 22, "query": "What were net sales in Americas region in 2023?", "target_section": ["americas", "segment information"], "target_year": "2023"},
    {"id": 23, "query": "What were net sales for iPhone in 2022?", "target_section": ["products", "segment information"], "target_year": "2022"},
    {"id": 24, "query": "What were net sales in Japan in 2022?", "target_section": ["japan", "segment information"], "target_year": "2022"},
    {"id": 25, "query": "What was operating income for Greater China in 2023?", "target_section": ["greater china", "segment information"], "target_year": "2023"},
    {"id": 26, "query": "What were net sales for Mac in 2022?", "target_section": ["segment information"], "target_year": "2022"},
    {"id": 27, "query": "What were net sales for Wearables Home and Accessories in 2023?", "target_section": ["segment information"], "target_year": "2023"},

    # --- Qualitative & Risk Factor Sections ---
    {"id": 28, "query": "What are main risk factors regarding international supply chain?", "target_section": ["item 1a. risk factors", "risk factors"], "target_year": None},
    {"id": 29, "query": "What are risks related to cybersecurity and data privacy?", "target_section": ["item 1a. risk factors", "risk factors"], "target_year": None},
    {"id": 30, "query": "What are risks related to intellectual property claims?", "target_section": ["item 1a. risk factors"], "target_year": None},
    {"id": 31, "query": "What are risks regarding foreign exchange rate fluctuations?", "target_section": ["risk factors", "quantitative and qualitative"], "target_year": None},
    {"id": 32, "query": "What are legal proceedings facing Apple?", "target_section": ["item 3. legal proceedings", "note 10"], "target_year": None},

    # --- Notes to Financial Statements & Other Complex Queries ---
    {"id": 33, "query": "What are lease related right of use assets in 2022?", "target_section": ["note 6 – leases", "leases"], "target_year": "2022"},
    {"id": 34, "query": "What were commercial paper outstanding balances in 2023?", "target_section": ["commercial paper"], "target_year": "2023"},
    {"id": 35, "query": "What were total non-current liabilities in 2022?", "target_section": ["balance sheets", "non-current liabilities"], "target_year": "2022"},
    {"id": 36, "query": "What are marketable securities balances in 2022?", "target_section": ["balance sheets", "financial instruments"], "target_year": "2022"},
    {"id": 37, "query": "What was inventory total value in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},
    {"id": 38, "query": "What were total non-current assets in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},
    {"id": 39, "query": "What were dividends paid per share in 2023?", "target_section": ["statements of cash flows", "capital stock"], "target_year": "2023"},
    {"id": 40, "query": "What was property plant and equipment net in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},

    # --- Additional Standard Checks ---
    {"id": 41, "query": "What were accounts receivable net in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},
    {"id": 42, "query": "What were vendor non trade receivables in 2022?", "target_section": ["consolidated balance sheets"], "target_year": "2022"},
    {"id": 43, "query": "What was total comprehensive income in 2021?", "target_section": ["statements of comprehensive income"], "target_year": "2021"},
    {"id": 44, "query": "What was net cash provided by investing activities in 2023?", "target_section": ["statements of cash flows"], "target_year": "2023"},
    {"id": 45, "query": "What was net cash used in financing activities in 2023?", "target_section": ["statements of cash flows"], "target_year": "2023"},
    {"id": 46, "query": "What were sales in Rest of Asia Pacific in 2022?", "target_section": ["rest of asia pacific", "segment information"], "target_year": "2022"},
    {"id": 47, "query": "What were interest and dividend income amounts in 2021?", "target_section": ["other income/(expense)", "note 5"], "target_year": "2021"},
    {"id": 48, "query": "What were term debt obligations as of 2022?", "target_section": ["term debt", "commercial paper"], "target_year": "2022"},
    {"id": 49, "query": "What are commitments and contingencies notes?", "target_section": ["note 10", "commitments"], "target_year": None},
    {"id": 50, "query": "What was total stockholders equity in 2022?", "target_section": ["balance sheets", "stockholders equity"], "target_year": "2022"}
]

# ============================================
# BENCHMARK EVALUATOR RUNNER
# ============================================

def evaluate_retriever():
    retriever = EnterpriseRetriever(INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME)
    
    print("\n" + "=" * 80)
    print("🚀 RUNNING FULL 50-QUESTION ENTERPRISE BENCHMARK EVALUATION 🚀")
    print("=" * 80 + "\n")

    top1_passes = 0
    top3_passes = 0
    total = len(TEST_DATASET)

    start_time = time.time()

    for item in TEST_DATASET:
        qid = item["id"]
        query = item["query"]
        target_sections = item["target_section"]
        target_year = item["target_year"]

        results = retriever.search(query, top_k=3)

        top1_match = False
        top3_match = False

        for rank, res in enumerate(results, start=1):
            chunk = res["chunk"]
            section = chunk.get("section", "").lower()
            text = chunk.get("text", "").lower()
            year = str(chunk.get("year", ""))

            # Matching criteria
            sec_hit = any(ts in section or ts in text for ts in target_sections)
            year_hit = (target_year is None) or (year == target_year)

            if sec_hit and year_hit:
                if rank == 1:
                    top1_match = True
                top3_match = True
                break

        if top1_match:
            top1_passes += 1
            print(f"✅ Q{qid:02d} [TOP 1 PASS] : '{query}'")
        elif top3_match:
            top3_passes += 1
            print(f"⚠️ Q{qid:02d} [TOP 3 PASS] : '{query}'")
        else:
            top1_chunk = results[0]["chunk"] if results else {}
            print(f"❌ Q{qid:02d} [FAILED]     : '{query}' | Got Section: [{top1_chunk.get('section', 'N/A')}] Year: [{top1_chunk.get('year', 'N/A')}]")

    elapsed = time.time() - start_time
    top1_accuracy = (top1_passes / total) * 100
    top3_accuracy = ((top1_passes + top3_passes) / total) * 100

    print("\n" + "=" * 80)
    print("📊 FINAL RETRIEVAL SYSTEM EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases Evaluation : {total}")
    print(f"Time Taken                  : {elapsed:.2f} seconds")
    print(f"🎯 Top-1 Retrieval Accuracy  : {top1_accuracy:.2f}%")
    print(f"🎯 Top-3 Retrieval Accuracy  : {top3_accuracy:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_retriever()