import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer

# ===========================================
# PATHS
# ===========================================

VECTOR_FOLDER = r"C:\Users\riaze\Desktop\TechCorp-Financial-Enterprise-RAG\apple_pipeline\vector_store"

INDEX_PATH = os.path.join(VECTOR_FOLDER, "apple.index")
METADATA_PATH = os.path.join(VECTOR_FOLDER, "metadata.pkl")

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Model Loaded!")

# ==========================================
# LOAD FAISS
# ==========================================

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"FAISS Vectors : {index.ntotal}")
print(f"Metadata      : {len(metadata)}")

# ==========================================
# KEYWORD BONUS
# ==========================================

def keyword_bonus(query, chunk):

    query = query.lower()

    section = chunk["section"].lower()
    parent = chunk["parent_section"].lower()
    text = chunk["text"].lower()

    score = 0

    for word in query.split():

        if len(word) < 3:
            continue

        if word in section:
            score += 0.25

        if word in parent:
            score += 0.20

        if word in text:
            score += 0.08

    # Strong boosts
    if "risk" in query and "risk factor" in parent:
        score += 0.80

    if "employee" in query and "employees" in section:
        score += 0.80

    if "business strategy" in query and "business strategy" in section:
        score += 0.80

    if "apple pay" in query and "apple pay" in section:
        score += 0.80

    if "products" in query and "products" in section:
        score += 0.50

    
    if ("competitor" in query or "competition" in query):

        if "competition" in section:
            score += 0.80

        if "competition" in parent:
            score += 0.60

    return score

# ==========================================
# SEARCH FUNCTION
# ==========================================

def search(query, top_k=5):

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)
    print("=" * 80)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve more candidates
    scores, indices = index.search(
        query_embedding.reshape(1, -1),
        80
    )

    query_words = set(query.lower().split())

    results = []

    for faiss_score, idx in zip(scores[0], indices[0]):

        chunk = metadata[idx]

        text = chunk["text"].strip()

        # Skip tiny/useless chunks
        if len(text) < 30:
            continue

        if text.lower() == "apple inc.":
            continue

        if text.lower() == "none.":
            continue

        # Skip noisy financial statement headers
        bad_sections = [

            "consolidated balance sheets",
            "consolidated statements of comprehensive income",
            "consolidated statements",
            "shares amount earnings income/(loss) equity",
            "item 16. form 10-k summary"

        ]

        if chunk["section"].lower() in bad_sections:
            continue

        bonus = keyword_bonus(query, chunk)

        # ----------------------------------------
        # Semantic overlap filtering
        # ----------------------------------------

        text_lower = chunk["text"].lower()
        section_lower = chunk["section"].lower()
        parent_lower = chunk["parent_section"].lower()

        overlap = 0

        for word in query_words:

            if len(word) < 3:
                continue

            if word in section_lower:
                overlap += 3

            elif word in parent_lower:
                overlap += 2

            elif word in text_lower:
                overlap += 1

        # Penalize chunks having almost no overlap
        if overlap == 0:
            continue

        elif overlap == 1:
            faiss_score -= 0.10

        elif overlap == 2:
            faiss_score += 0.05

        elif overlap >= 3:
            faiss_score += 0.15

        final_score = float(faiss_score) + bonus

        results.append({
            "score": final_score,
            "faiss_score": float(faiss_score),
            "chunk": chunk
        })

    # Re-rank
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    results = results[:top_k]

    # Print results
    for rank, item in enumerate(results, start=1):

        chunk = item["chunk"]

        print("\n" + "-" * 80)
        print(f"Rank        : {rank}")
        print(f"Final Score : {item['score']:.4f}")
        print(f"FAISS Score : {item['faiss_score']:.4f}")
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Company     : {chunk['company']}")
        print(f"Year        : {chunk['year']}")
        print(f"Page        : {chunk['page']}")
        print(f"Parent      : {chunk['parent_section']}")
        print(f"Section     : {chunk['section']}")
        print("\nTEXT:\n")
        print(chunk["text"][:900])
        print("-" * 80)

# ==========================================
# INTERACTIVE LOOP
# ==========================================

while True:

    query = input("\nAsk Question (or type exit): ")

    if query.lower() == "exit":
        break

    search(query)