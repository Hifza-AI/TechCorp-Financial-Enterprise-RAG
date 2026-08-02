import os
import re
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION & PATHS
# ============================================

VECTOR_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\vector_store"
INDEX_PATH = os.path.join(VECTOR_FOLDER, "apple.index")
METADATA_PATH = os.path.join(VECTOR_FOLDER, "metadata.pkl")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K_CANDIDATES = 80
FINAL_TOP_K = 5

# ============================================
# RAG RETRIEVAL ENGINE CLASS
# ============================================

class EnterpriseRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        
        print("Loading Vector Index & Metadata...")
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        print(f"FAISS Vectors : {self.index.ntotal}")
        print(f"Metadata      : {len(self.metadata)}")

    def _calculate_lexical_overlap(self, query: str, chunk: Dict[str, Any]) -> float:
        query_lower = query.lower()
        section = chunk.get("section", "").lower()
        parent = chunk.get("parent_section", "").lower()
        text = chunk.get("text", "").lower()
        chunk_year = str(chunk.get("year", ""))

        score = 0.0

        # --- 1. YEAR MATCHING LOGIC (DYNAMIC METADATA FILTER) ---
        extracted_years = re.findall(r'\b(20\d{2})\b', query_lower)
        if extracted_years:
            target_year = extracted_years[0]
            if chunk_year == target_year:
                score += 0.25  # Year Match Boost!

        # --- 2. EXACT PHRASE MATCHING ---
        if "net income" in query_lower:
            if "consolidated statements" in section or "comprehensive income" in section or "statements of operations" in section:
                score += 0.80
            if "net income" in text:
                score += 0.40

        # --- 3. WORD LEVEL MATCHING ---
        query_words = set(query_lower.split())
        for word in query_words:
            if len(word) < 3 or word in ["what", "was", "the", "in", "and", "for"]:
                continue

            if word in section:
                score += 0.05
            elif word in text:
                score += 0.01

        return score

    def search(self, query: str, top_k: int = FINAL_TOP_K) -> List[Dict[str, Any]]:
        # 1. Generate Embeddings
        query_embedding = self.model.encode(
            query, 
            normalize_embeddings=True
        ).astype("float32").reshape(1, -1)

        # 2. Dense Vector Retrieval
        scores, indices = self.index.search(query_embedding, TOP_K_CANDIDATES)

        results = []

        # 3. Dynamic Re-ranking & Filtering
        for faiss_score, idx in zip(scores[0], indices[0]):
            chunk = self.metadata[idx]
            text = chunk.get("text", "").strip()

            if len(text) < 30 or text.lower() in ["apple inc.", "none."]:
                continue

            # Pass full string 'query' here
            lexical_score = self._calculate_lexical_overlap(query, chunk)

            # Combined Score Hybrid Formula
            final_score = float(faiss_score) + lexical_score

            results.append({
                "score": final_score,
                "vector_score": float(faiss_score),
                "lexical_score": lexical_score,
                "chunk": chunk
            })

        # 4. Sort Candidates
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# ============================================
# ACCURACY EVALUATION / TEST BENCHMARK
# ============================================

def run_accuracy_benchmark(retriever: EnterpriseRetriever):
    """
    Evaluates top retrieval accuracy across different standard queries.
    """
    test_cases = [
        {
            "query": "What was the net income in 2021?",
            "expected_keywords": ["consolidated statements of comprehensive income", "net income"],
            "expected_year": "2021"
        },
        {
            "query": "What were the research and development expenses in 2024?",
            "expected_keywords": ["research and development", "r&d"],
            "expected_year": "2024"
        },
        {
            "query": "What are Apple's segment net sales for Greater China?",
            "expected_keywords": ["greater china", "segment"],
            "expected_year": None
        },
        {
            "query": "What were total current assets in 2021?",
            "expected_keywords": ["balance sheet", "current assets"],
            "expected_year": "2021"
        }
    ]

    print("\n" + "=" * 80)
    print("🚀 RUNNING AUTOMATED RAG ACCURACY TEST BENCHMARK 🚀")
    print("=" * 80)

    passed = 0
    total = len(test_cases)

    for idx, test in enumerate(test_cases, 1):
        results = retriever.search(test["query"], top_k=3)
        top_chunk = results[0]["chunk"] if results else {}
        top_text = top_chunk.get("text", "").lower()
        top_section = top_chunk.get("section", "").lower()
        top_year = str(top_chunk.get("year", ""))

        # Validation Logic
        keyword_match = any(kw in top_text or kw in top_section for kw in test["expected_keywords"])
        year_match = (test["expected_year"] is None) or (top_year == test["expected_year"])

        if keyword_match and year_match:
            print(f"✅ Test {idx}: PASSED | Query: '{test['query']}' | Top Section: [{top_section}] Year: [{top_year}]")
            passed += 1
        else:
            print(f"❌ Test {idx}: FAILED | Query: '{test['query']}' | Retrieved: [{top_section}] Year: [{top_year}]")

    accuracy = (passed / total) * 100
    print("-" * 80)
    print(f"📊 SYSTEM RETRIEVAL ACCURACY: {accuracy:.2f}%\n" + "=" * 80)

# ============================================
# EXECUTION PIPELINE
# ============================================

if __name__ == "__main__":
    retriever = EnterpriseRetriever(INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME)

    # Automatically run accuracy test first!
    run_accuracy_benchmark(retriever)

    while True:
        user_query = input("\nAsk Question (or type benchmark / exit): ").strip()

        if user_query.lower() in ["exit", "quit"]:
            break

        if user_query.lower() == "benchmark":
            run_accuracy_benchmark(retriever)
            continue

        if not user_query:
            continue

        search_results = retriever.search(user_query)

        print("\n" + "=" * 80)
        print(f"QUERY: {user_query}")
        print("=" * 80)

        for rank, item in enumerate(search_results, start=1):
            chunk = item["chunk"]
            print(f"\nRank        : {rank}")
            print(f"Final Score : {item['score']:.4f} (Vector: {item['vector_score']:.4f} | Lexical: {item['lexical_score']:.4f})")
            print(f"Chunk ID    : {chunk.get('chunk_id', 'N/A')}")
            print(f"Company     : {chunk.get('company', 'N/A')}")
            print(f"Year        : {chunk.get('year', 'N/A')}")
            print(f"Section     : {chunk.get('section', 'N/A')}")
            print("\nTEXT SNIPPET:\n")
            print(chunk["text"][:600] + "...")
            print("-" * 80)