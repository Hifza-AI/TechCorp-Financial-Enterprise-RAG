import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Offline mode enforce karein
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("🔍 Searching for FAISS index and metadata files in project...")

# Dynamically find .index and metadata file
index_path = None
meta_path = None

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".index") and not index_path:
            index_path = os.path.join(root, file)
        if ("metadata" in file.lower() or "meta" in file.lower()) and file.endswith(".pkl") and not meta_path:
            meta_path = os.path.join(root, file)

print(f"✅ Found Index File:    {index_path}")
print(f"✅ Found Metadata File: {meta_path}")

if not index_path or not meta_path:
    print("\n❌ Error: Could not automatically locate .index or .pkl files!")
    print("Available files in directory:")
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(('.index', '.pkl', '.json')):
                print(f"  - {os.path.join(root, f)}")
    exit(1)

print("\nLoading model and FAISS index...")
try:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
except Exception:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = faiss.read_index(index_path)

with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

# 15 Failed Queries
failed_queries = [
    (5, "What was total current liabilities in 2021?"),
    (8, "What was total operating income in 2021?"),
    (16, "What were depreciation and amortization costs in 2023?"),
    (17, "What were share based compensation expenses in 2023?"),
    (26, "What were net sales for Mac in 2022?"),
    (27, "What were net sales for Wearables Home and Accessories in 2023?"),
    (29, "What are risks related to cybersecurity and data privacy?"),
    (30, "What are risks related to intellectual property claims?"),
    (31, "What are risks regarding foreign exchange rate fluctuations?"),
    (32, "What are legal proceedings facing Apple?"),
    (37, "What was inventory total value in 2022?"),
    (39, "What were dividends paid per share in 2023?"),
    (41, "What were accounts receivable net in 2022?"),
    (42, "What were vendor non trade receivables in 2022?"),
    (50, "What was total stockholders equity in 2022?")
]

print("\n" + "="*80)
print("🔍 MANUAL AUDIT LOG: RETRIEVED TEXT FOR FAILED QUERIES")
print("="*80 + "\n")

for q_num, q_text in failed_queries:
    instruction_query = f"Represent this sentence for searching relevant passages: {q_text}"
    q_emb = model.encode([instruction_query], normalize_embeddings=True)
    
    # Retrieve top 2 chunks
    D, I = index.search(q_emb, 2)
    
    print(f"📌 Q{q_num:02d}: '{q_text}'")
    for rank, idx in enumerate(I[0]):
        meta = metadata[idx]
        sec = meta.get('section', 'N/A')
        yr = meta.get('year', 'N/A')
        text_preview = meta.get('text', '')[:180].replace('\n', ' ')
        print(f"   Rank {rank+1} -> [Year: {yr}] [Section: {sec}]")
        print(f"           Preview: \"{text_preview}...\"")
    print("-" * 80)