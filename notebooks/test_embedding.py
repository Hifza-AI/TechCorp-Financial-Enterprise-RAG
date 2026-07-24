from sentence_transformers import SentenceTransformer
import os

# -------------------------
# Step 1: Load all chunks
# -------------------------

chunks_folder = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\data\chunks\Apple"

all_chunks = []

for report_folder in os.listdir(chunks_folder):
    report_path = os.path.join(chunks_folder, report_folder)

    if os.path.isdir(report_path):

        for file in sorted(os.listdir(report_path)):
            if file.endswith(".txt"):

                with open(os.path.join(report_path, file), "r", encoding="utf-8") as f:
                    all_chunks.append(f.read())

print(f"\nTotal Chunks Loaded: {len(all_chunks)}")

# -------------------------
# Step 2: Load Sentence Transformer
# -------------------------

print("\nLoading Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully!")

# -------------------------
# Step 3: Generate embeddings
# -------------------------

sample_chunks = all_chunks[:5]

embeddings = model.encode(sample_chunks)

print("\nEmbeddings Generated Successfully!")

print("\nEmbedding Shape:")

print(embeddings.shape)

print("\nFirst Embedding (first 10 values):")

print(embeddings[0][:10])