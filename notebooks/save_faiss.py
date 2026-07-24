from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
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

                with open(
                    os.path.join(report_path, file),
                    "r",
                    encoding="utf-8"
                ) as f:

                    all_chunks.append(f.read())

print(f"Total Chunks Loaded: {len(all_chunks)}")

# -----------------------------
# Generate Embeddings
# -----------------------------
print("\nGenerating Embeddings...")

embeddings = model.encode(
    all_chunks,
    batch_size=32,
    show_progress_bar=True
)

print("Embeddings Generated!")

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("FAISS Index Ready!")

# -----------------------------
# Save Everything
# -----------------------------
save_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\vector_store"

os.makedirs(save_folder, exist_ok=True)

# Save FAISS
faiss.write_index(
    index,
    os.path.join(save_folder, "apple_index.faiss")
)

# Save Embeddings
np.save(
    os.path.join(save_folder, "apple_embeddings.npy"),
    embeddings
)

# Save Metadata
with open(
    os.path.join(save_folder, "apple_metadata.pkl"),
    "wb"
) as f:

    pickle.dump(all_chunks, f)

print("\nEverything Saved Successfully!")