from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import pickle
import numpy as np

# ======================================================
# Load Model
# ======================================================

print("Loading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!\n")

# ======================================================
# Load FAISS
# ======================================================

index = faiss.read_index(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_index.faiss"
)

print("FAISS Loaded!")

# ======================================================
# Load Metadata
# ======================================================

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_metadata_v3.pkl",
    "rb"
) as f:

    metadata = pickle.load(f)

print("Metadata Loaded!")

# ======================================================
# Load BM25
# ======================================================

with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_bm25.pkl",
    "rb"
) as f:

    bm25 = pickle.load(f)

print("BM25 Loaded!\n")

# ======================================================
# Query
# ======================================================

query = "What was Apple's total revenue in 2022?"

print("="*80)
print("User Query:")
print(query)

# ======================================================
# FAISS Search
# ======================================================

query_embedding = model.encode([query])

faiss_distances, faiss_indices = index.search(query_embedding, 10)

# ======================================================
# BM25 Search
# ======================================================

tokens = query.lower().split()

bm25_scores = bm25.get_scores(tokens)

top_bm25 = np.argsort(bm25_scores)[::-1][:10]

# ======================================================
# Merge Results
# ======================================================

merged = []

for idx in faiss_indices[0]:
    merged.append(idx)

for idx in top_bm25:
    if idx not in merged:
        merged.append(idx)

# ======================================================
# Print Results
# ======================================================

print("\nHybrid Retrieved Chunks\n")

for rank, idx in enumerate(merged[:10]):

    print("="*80)
    print(f"Rank {rank+1}")

    print(f"Company : {metadata[idx]['company']}")
    print(f"Year    : {metadata[idx]['year']}")
    print(f"Section : {metadata[idx]['section']}")
    print(f"Chunk   : {metadata[idx]['chunk_id']}")

    print("\nPreview:\n")

    print(metadata[idx]["text"][:1000])

    print()