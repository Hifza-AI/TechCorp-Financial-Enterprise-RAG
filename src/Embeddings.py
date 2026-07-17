# Create Embeddings

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentence = "Googgle got loss in 2026"

embedding = model.encode(sentence)

print(embedding)

print(len(embedding))

# Calculating Cosine Similarity 


from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
             "Google revenue increased in 2024." ,
             "Google earnings increased last year." ,
             "Apple launched a new IPhone"

]

embeddings = model.encode(sentences)
print(cosine_similarity(embeddings))


