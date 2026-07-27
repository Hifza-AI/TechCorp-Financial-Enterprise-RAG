import pickle
import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =====================================================
# PATHS
# =====================================================

FAISS_PATH = r"data/vector_store/apple_index.faiss"
METADATA_PATH = r"data/vector_store/apple_metadata_v3.pkl"
BM25_PATH = r"data/vector_store/apple_bm25.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =====================================================
# LOAD EVERYTHING
# =====================================================

print("Loading Embedding Model...")
embedding_model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS...")
faiss_index = faiss.read_index(FAISS_PATH)

print("Loading Metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading BM25...")
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

print("Retriever Ready!\n")


# =====================================================
# METADATA FILTER
# =====================================================

def metadata_filter(company=None, year=None, section=None):

    filtered = []

    for idx, item in enumerate(metadata):

        if company is not None:
            if item["company"].lower() != company.lower():
                continue

        if year is not None:
            if str(item["year"]) != str(year):
                continue

        if section is not None:
            if item["section"].lower() != section.lower():
                continue

        filtered.append((idx, item))

    return filtered


# =====================================================
# BM25 SEARCH
# =====================================================

def bm25_search(query, filtered_chunks, top_k=20):

    if len(filtered_chunks) == 0:
        return []

    corpus = []

    mapping = []

    for idx, item in filtered_chunks:

        corpus.append(item["text"].split())
        mapping.append(idx)

    local_bm25 = BM25Okapi(corpus)

    scores = local_bm25.get_scores(query.split())

    order = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in order:

        results.append({

            "index": mapping[i],

            "score": float(scores[i]),

            "metadata": metadata[mapping[i]]

        })

    return results


# =====================================================
# FAISS SEARCH
# =====================================================

def faiss_search(query, filtered_chunks, top_k=20):

    if len(filtered_chunks) == 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    D, I = faiss_index.search(query_embedding, 100)

    allowed = set(idx for idx, _ in filtered_chunks)

    results = []

    for score, idx in zip(D[0], I[0]):

        if idx in allowed:

            results.append({

                "index": idx,

                "score": float(score),

                "metadata": metadata[idx]

            })

        if len(results) >= top_k:
            break

    return results


# =====================================================
# RECIPROCAL RANK FUSION
# =====================================================

def hybrid_search(query,
                  company=None,
                  year=None,
                  section=None,
                  top_k=10):

    filtered = metadata_filter(
        company=company,
        year=year,
        section=section
    )

    bm25_results = bm25_search(query, filtered, top_k=30)

    faiss_results = faiss_search(query, filtered, top_k=30)

    rrf_scores = {}

    k = 60

    for rank, item in enumerate(bm25_results):

        idx = item["index"]

        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, item in enumerate(faiss_results):

        idx = item["index"]

        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    final = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    output = []

    for idx, score in final:

        output.append({

            "score": score,

            "metadata": metadata[idx]

        })

    return output