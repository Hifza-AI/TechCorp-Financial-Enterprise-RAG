from gensim.models import FastText

sentences = [
    ["google","revenue","increased"],
    ["sales","grew"],
    ["profit","rose"]
]

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

print(model.wv["Hifza"])