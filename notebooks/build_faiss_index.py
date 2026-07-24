from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# -----------------------------
# Load Model
# -----------------------------
print("Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!\n")

# -----------------------------
# Load Chunks
# -----------------------------
chunks_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks\Apple"

all_chunks = []

for report in os.listdir(chunks_folder):

    report_path = os.path.join(chunks_folder, report)

    if os.path.isdir(report_path):

        for file in os.listdir(report_path):

            if file.endswith(".txt"):

                with open(os.path.join(report_path, file),
                          "r",
                          encoding="utf-8") as f:

                    all_chunks.append(f.read())

print(f"Total Chunks Loaded: {len(all_chunks)}")

# -----------------------------
# Generate Embeddings
# -----------------------------
print("\nGenerating Embeddings...")

embeddings = model.encode(
    all_chunks,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings Generated!")

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS Index Created!")

print("Total vectors stored:", index.ntotal)