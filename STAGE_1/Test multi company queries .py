"""
test_multi_company_queries.py

Runs a focused set of test queries -- across ALL 15 indexed companies
(Apple, Google, Intel, Meta, Nvidia, Adobe, Amazon, Chipotle, Netflix,
ServiceNow, Palo Alto Networks, Salesforce, Starbucks, Walmart,
Workday) -- against the Retriever, and prints the TOP result for each
so you can quickly eyeball whether it's correct (right numbers, right
section) against the real PDF.

NEW: extended from the original 5-company version to cover all 15
companies now indexed. The ORIGINAL 5 companies' groups (A-H) are
UNCHANGED -- same wording, same explicit years ("2024", "fiscal
2025" for Nvidia) that were already proven to work, so any accuracy
comparison against the earlier baseline stays apples-to-apples.

For the 10 NEWLY added companies, queries deliberately use
"most recent"/"latest" phrasing rather than a specific year number.
This is a safety choice, not a style preference: these 10 companies
have SIX different fiscal-year-end conventions (calendar year;
late-September/early-October for Starbucks; July 31 for Palo Alto
Networks; January 31 for Salesforce, Walmart, and Workday), so a
guessed explicit year risks silently testing the WRONG year for a
given company purely due to a wording mismatch -- a false test
failure that has nothing to do with the retriever itself. Recency
phrasing resolves against the corpus's own actual metadata instead,
so it can't be wrong for this reason.

A few company-specific queries are included where this project's own
review already confirmed a distinctive, worth-checking structure --
e.g. Starbucks' negative shareholders' equity (deficit), Workday's
dual-class (Class A / Class B) stock, and Chipotle's single
reportable segment.

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
    # ORIGINAL 5 companies -- unchanged from the proven baseline
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

    # ---- NEW 10 companies -- recency-phrased, self-resolving ----
    "What was Adobe's most recent total revenue?",
    "What was Adobe's most recent net income?",
    "What was Amazon's most recent total net sales?",
    "What was Amazon's most recent net income?",
    "What was Chipotle's most recent total revenue?",
    "What was Chipotle's most recent net income?",
    "What was Netflix's most recent total revenue?",
    "What was Netflix's most recent net income?",
    "What was ServiceNow's most recent total revenue?",
    "What was Palo Alto Networks' most recent total revenue?",
    "What was Salesforce's most recent total revenue?",
    "What was Starbucks' most recent total net revenues?",
    "What was Walmart's most recent total revenue?",
    "What was Workday's most recent total revenue?",
    "What was Workday's most recent net income?",

    # =====================================================
    # Group B -- table-based lookups
    # ORIGINAL 5 companies -- unchanged
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

    # ---- NEW 10 companies ----
    "What was Adobe's total assets on the balance sheet?",
    "What was Amazon's total assets on the balance sheet?",
    "What was Chipotle's total assets on the balance sheet?",
    "What was Netflix's total assets on the balance sheet?",
    "What was ServiceNow's total assets on the balance sheet?",
    "What was Palo Alto Networks' total stockholders' equity?",
    "What was Salesforce's total stockholders' equity?",
    "What was Starbucks' total shareholders' equity or deficit?",
    "What was Walmart's total assets on the balance sheet?",
    "What was Workday's total stockholders' equity?",

    # =====================================================
    # Group C -- geographic / segment
    # ORIGINAL 5 companies -- unchanged
    # =====================================================
    "How much revenue did Apple generate in Greater China?",
    "What were Google's revenues by geography?",
    "What were Intel's operating segments?",
    "What is Meta's Family of Apps revenue?",
    "What was Nvidia's Data Center segment revenue?",

    # ---- NEW 10 companies ----
    "What are Adobe's reportable segments?",
    "What were Amazon's segment operating results?",
    "Does Chipotle operate as a single reportable segment?",
    "What is Netflix's revenue by region?",
    "What are ServiceNow's reportable segments?",
    "What are Palo Alto Networks' reportable segments?",
    "What are Salesforce's reportable operating segments?",
    "What are Starbucks' three reportable operating segments?",
    "What are Walmart's three reportable segments?",
    "What is Workday's revenue by geography?",

    # =====================================================
    # Group D -- Notes-specific
    # ORIGINAL 5 companies -- unchanged
    # =====================================================
    "What is Apple's revenue recognition policy?",
    "What are Apple's deferred tax assets?",
    "What is Google's stock-based compensation expense?",
    "What is Intel's goodwill balance?",
    "What is Meta's lease liability maturity schedule?",
    "What is Nvidia's inventory valuation policy?",

    # ---- NEW 10 companies ----
    "What is Adobe's revenue recognition policy?",
    "What is Amazon's stock-based compensation expense?",
    "What is Chipotle's lease accounting policy?",
    "What is Netflix's content assets accounting policy?",
    "What is ServiceNow's deferred revenue balance?",
    "What is Palo Alto Networks' goodwill balance?",
    "What is Salesforce's business combinations and acquisitions activity?",
    "What is Starbucks' stored value card and loyalty program liability?",
    "What is Walmart's long-term debt maturity schedule?",
    "What is Workday's deferred commissions accounting policy?",

    # =====================================================
    # Group E -- Risk Factors
    # ORIGINAL 5 companies -- unchanged
    # =====================================================
    "What risks does Apple face related to product introductions and transitions?",
    "What risks does Google disclose related to competition?",
    "What risks does Intel disclose related to manufacturing?",
    "What competitive risks does Meta mention?",
    "What supply chain risks does Nvidia disclose?",

    # ---- NEW 10 companies ----
    "What risks does Adobe disclose related to AI?",
    "What risks does Amazon disclose related to international operations?",
    "What risks does Chipotle disclose related to food safety?",
    "What risks does Netflix disclose related to content costs?",
    "What risks does ServiceNow disclose related to data centers and third-party infrastructure?",
    "What risks does Palo Alto Networks disclose related to cybersecurity?",
    "What risks does Salesforce disclose related to AI and Agentforce?",
    "What risks does Starbucks disclose related to labor costs?",
    "What risks does Walmart disclose related to store growth?",
    "What risks does Workday disclose related to data center disruptions?",

    # =====================================================
    # Group F -- recency-aware
    # ORIGINAL companies -- unchanged
    # =====================================================
    "What is Apple's most recent total net sales?",
    "What is Google's latest reported revenue?",
    "What is Nvidia's most recent quarterly revenue?",

    # ---- NEW companies ----
    "What is Adobe's latest reported net income?",
    "What is Chipotle's latest reported revenue?",
    "What is Starbucks' latest reported net earnings?",

    # =====================================================
    # Group G -- cross-company differentiation (company-filter sanity check)
    # =====================================================
    "What was Google's revenue?",       # should NOT return Apple data
    "What was Intel's net income?",     # should NOT return Meta data
    "What was Chipotle's revenue?",     # should NOT return Starbucks data (both restaurants)
    "What was Salesforce's revenue?",   # should NOT return ServiceNow data (both enterprise SaaS)
    "What was Walmart's revenue?",      # should NOT return Amazon data (both retail)

    # =====================================================
    # Group H -- company-specific structural checks
    # (distinctive patterns this project's own review already
    # confirmed are worth specifically verifying retrieve correctly)
    # =====================================================
    "Does Starbucks have a shareholders' deficit?",
    "Does Workday have Class A and Class B common stock?",
    "What is Palo Alto Networks' fiscal year end date?",
    "What is Walmart's fiscal year end date?",

    # =====================================================
    # Group I -- out-of-scope (should NOT confidently match)
    # =====================================================
    "What is Apple's current stock price today?",
    "Who is the CEO of Tesla?",
    "What is Microsoft's total revenue?",
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