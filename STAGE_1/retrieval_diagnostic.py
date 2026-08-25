import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

VECTOR_STORE = Path("STAGE_1/vector_store")
INDEX_PATH = VECTOR_STORE / "index.faiss"
METADATA_PATH = VECTOR_STORE / "metadata.pkl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "company": "Apple",
        "query": "What was Apple's total assets on the balance sheet?",
        "expected_terms": ["total assets"],
        "expected_sections": ["balance sheet", "financial statements"],
    },
    {
        "company": "Microsoft",
        "query": "What was Microsoft's total liabilities?",
        "expected_terms": ["total liabilities"],
        "expected_sections": ["balance sheet", "financial statements"],
    },
    {
        "company": "CVS",
        "query": "What was CVS's earnings per share (EPS)?",
        "expected_terms": [
            "earnings per share",
            "diluted earnings per share",
        ],
        "expected_sections": [
            "statement of operations",
            "financial statements",
            "earnings per share",
        ],
    },
    {
        "company": "Costco",
        "query": "What was Costco's operating income?",
        "expected_terms": ["operating income"],
        "expected_sections": [
            "financial statements",
            "selected financial data",
            "income statement",
        ],
    },
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"FAISS vectors : {index.ntotal}")
print(f"Metadata rows : {len(metadata)}")

print(f"\nLoading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# ============================================================
# HELPER
# ============================================================

def normalize(value):
    return str(value).lower().strip()


def is_expected_chunk(chunk, test):
    """
    Determines whether a chunk looks like the expected
    financial answer chunk.

    This is NOT ground truth.
    It is only a diagnostic heuristic.
    """

    company = normalize(chunk.get("company", ""))
    text = normalize(chunk.get("text", ""))
    section = normalize(chunk.get("section_path", ""))
    chunk_type = normalize(chunk.get("chunk_type", ""))

    if normalize(test["company"]) not in company:
        return False

    # At least one exact financial metric phrase
    term_match = any(
        term.lower() in text
        for term in test["expected_terms"]
    )

    if not term_match:
        return False

    # Financial/table sections are preferred
    section_match = any(
        section_term in section
        for section_term in test["expected_sections"]
    )

    # Tables containing the metric are especially useful
    if chunk_type == "table" and term_match:
        return True

    if section_match and term_match:
        return True

    return False


# ============================================================
# DIAGNOSTIC
# ============================================================

def run_diagnostic(test, search_k=100):
    query = test["query"]

    print("\n" + "=" * 90)
    print("QUERY")
    print("=" * 90)
    print(query)

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        min(search_k, index.ntotal),
    )

    scores = scores[0]
    indices = indices[0]

    # --------------------------------------------------------
    # Find expected chunks globally
    # --------------------------------------------------------

    expected_indices = []

    for i, chunk in enumerate(metadata):
        if is_expected_chunk(chunk, test):
            expected_indices.append(i)

    print("\nExpected-like chunks found in metadata:")
    print(len(expected_indices))

    if not expected_indices:
        print(
            "\nWARNING: No expected-like chunk found "
            "using the current diagnostic rules."
        )
        return

    # --------------------------------------------------------
    # Find their FAISS ranks
    # --------------------------------------------------------

    rank_results = []

    for expected_index in expected_indices:

        rank = None
        score = None

        for position, faiss_index in enumerate(indices):

            if int(faiss_index) == expected_index:
                rank = position + 1
                score = float(scores[position])
                break

        rank_results.append(
            {
                "metadata_index": expected_index,
                "rank": rank,
                "score": score,
                "chunk": metadata[expected_index],
            }
        )

    # --------------------------------------------------------
    # Sort:
    # Found chunks first, then by rank
    # --------------------------------------------------------

    rank_results.sort(
        key=lambda x: (
            x["rank"] is None,
            x["rank"] if x["rank"] is not None else 999999,
        )
    )

    # --------------------------------------------------------
    # Print best expected chunks
    # --------------------------------------------------------

    print("\nBEST EXPECTED CHUNKS")
    print("-" * 90)

    shown = 0

    for result in rank_results:

        if shown >= 10:
            break

        chunk = result["chunk"]

        print(f"\nMetadata index : {result['metadata_index']}")

        if result["rank"] is None:
            print("FAISS rank     : NOT IN TOP-K")
            print("FAISS score    : ---")
        else:
            print(f"FAISS rank     : #{result['rank']}")
            print(f"FAISS score    : {result['score']:.4f}")

        print(f"Year           : {chunk.get('year')}")
        print(f"Type           : {chunk.get('chunk_type')}")
        print(f"Section        : {chunk.get('section_path')}")

        print("\nText preview:")
        print(str(chunk.get("text", ""))[:800])

        shown += 1

    # --------------------------------------------------------
    # Print actual top 10 FAISS results
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ACTUAL FAISS TOP 10")
    print("=" * 90)

    for position in range(min(10, len(indices))):

        idx = int(indices[position])

        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]

        print(
            f"\n[{position + 1}] "
            f"score={float(scores[position]):.4f} | "
            f"{chunk.get('company')} | "
            f"{chunk.get('year')} | "
            f"{chunk.get('chunk_type')}"
        )

        print(
            f"Section: "
            f"{chunk.get('section_path')}"
        )

        print(
            f"Text: "
            f"{str(chunk.get('text', ''))[:500]}"
        )


# =============================================================
# RUN ALL TESTS
# =============================================================

if __name__ == "__main__":

    for test in TEST_CASES:
        run_diagnostic(test, search_k=100)

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)