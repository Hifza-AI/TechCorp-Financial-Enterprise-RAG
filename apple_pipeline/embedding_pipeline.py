import os
import json
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# PATHS
# ==========================================

CHUNK_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\chunks"

OUTPUT_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\vector_store"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Model Loaded!")

# ==========================================
# LOAD CHUNKS
# ==========================================

all_embeddings = []
all_metadata = []

chunk_files = sorted(os.listdir(CHUNK_FOLDER))

for file in chunk_files:

    if not file.endswith(".json"):
        continue

    print(f"Processing {file}")

    with open(os.path.join(CHUNK_FOLDER, file), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:

        text = chunk["text"]

        embedding = model.encode(
            text,
            normalize_embeddings=True
        )

        all_embeddings.append(embedding.astype("float32"))

        metadata = chunk.copy()

        all_metadata.append(metadata)

print(f"\nTotal Chunks : {len(all_metadata)}")

# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = len(all_embeddings[0])

index = faiss.IndexFlatIP(dimension)

embeddings_np = np.array(all_embeddings).astype("float32")

index.add(embeddings_np)

print("FAISS Index Created!")

# ==========================================
# SAVE
# ==========================================

faiss.write_index(
    index,
    os.path.join(OUTPUT_FOLDER, "apple.index")
)

with open(
    os.path.join(OUTPUT_FOLDER, "metadata.pkl"),
    "wb"
) as f:

    pickle.dump(all_metadata, f)

print("\n===================================")
print("Embedding Pipeline Finished!")
print("===================================")
print(f"Vectors : {index.ntotal}")
print("Saved Index")
print("Saved Metadata")