from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# -----------------------------
# Load Model
# -----------------------------
print("Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!\n")

# -----------------------------
# Load Saved FAISS Index
# -----------------------------
index = faiss.read_index(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_index.faiss"
)

print("FAISS Loaded!")

# -----------------------------
# Load Metadata
# -----------------------------
with open(
    r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_metadata.pkl",
    "rb"
) as f:

    all_chunks = pickle.load(f)

print("Metadata Loaded!")

# -----------------------------
# User Query
# -----------------------------
query = "What was Apple's total revenue in 2022?"

print("\nUser Query:")
print(query)

# -----------------------------
# Query Embedding
# -----------------------------
query_embedding = model.encode([query])

# -----------------------------
# Search
# -----------------------------
k = 5

distances, indices = index.search(query_embedding, k)

# -----------------------------
# Show Results
# -----------------------------
print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):

    print("=" * 80)
    print(f"Rank {rank+1}")
    print(f"Chunk Index: {idx}")
    print(f"Distance: {distances[0][rank]}")

    print("\nChunk Preview:\n")

    print(all_chunks[idx][:1200])

    print()