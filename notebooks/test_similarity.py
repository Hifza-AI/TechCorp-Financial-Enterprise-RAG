from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Apple revenue increased in 2021.",
    "Apple earned more money in 2021.",
    "Microsoft acquired GitHub."
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

print("\nCosine Similarity Matrix:\n")
print(similarity)