import os
import pickle
from rank_bm25 import BM25Okapi

# ==============================
# Load Metadata
# ==============================

metadata_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_metadata_v3.pkl"

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

print("Metadata Loaded!")

# ==============================
# Tokenize
# ==============================

tokenized_corpus = []

for item in metadata:
    tokens = item["text"].lower().split()
    tokenized_corpus.append(tokens)

# ==============================
# Build BM25
# ==============================

bm25 = BM25Okapi(tokenized_corpus)

# ==============================
# Save
# ==============================

save_path = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\vector_store\apple_bm25.pkl"

with open(save_path, "wb") as f:
    pickle.dump(bm25, f)

print("="*50)
print("BM25 Saved Successfully")
print("="*50)

print("Total Documents :", len(tokenized_corpus))