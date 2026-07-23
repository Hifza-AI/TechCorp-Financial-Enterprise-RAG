from transformers import pipeline

classifier = pipeline(
"sentiment-analysis")
result = classifier("I Love AI.")
print(result)