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
queries = [
    "What was Apple's total revenue in 2022?",
    "What was Apple's net income in 2022?",
    "How much revenue came from iPhone?",
    "What are Apple's services?",
    "What risks did Apple mention?",
    "What was Apple's cash balance?",
]


# -----------------------------
# Query Embedding
# -----------------------------

# -----------------------------
# Search
# -----------------------------

query_embeddings = model.encode(queries)

k = 5

for i, query in enumerate(queries):

    print("="*100)
    print("User Query:")
    print(query)

    distances, indices = index.search(
        query_embeddings[i].reshape(1,-1),
        k
    )

    print("\nTop Results:\n")

    for rank, idx in enumerate(indices[0]):

        print("="*80)
        print(f"Rank {rank+1}")
        print(f"Chunk Index: {idx}")
        print(f"Distance: {distances[0][rank]}")

        print("\nChunk Preview:\n")

        print(all_chunks[idx][:1200])

        print()