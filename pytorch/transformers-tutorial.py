# transformers-tutorial.py

"""
Code for the HuggingFace tutorial found here:
https://huggingface.co/course

For now the working virtual environment is in ./pytorch

"""

from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))


sentiment_probabiltiy = classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
])

print("The sentiment probability is: {sentiment_probability}")

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
probabilities = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(f'The probabilities are {probabilities}')