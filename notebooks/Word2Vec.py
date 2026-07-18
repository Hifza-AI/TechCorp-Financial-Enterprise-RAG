from gensim.models import Word2Vec

sentences = [
    ["google", "revenue", "increased"],
    ["sales", "grew"],
    ["profit", "rose"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

print(model.wv["revenue"])