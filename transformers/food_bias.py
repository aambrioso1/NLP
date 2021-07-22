# food_basis.py

# Sentiment analysis

# Start Chapter 2 here:  https://huggingface.co/course/chapter2/3?fw=pt

from transformers import pipeline


classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

food_list = ["hamburger", "salad", "cake", "yogurt", "beets", "red meat",\
"healthy food", "fattening food", "candy", "liver", "fruit", "insect",\
 "dog", "horse", "man", "monkey", "food high in calories", "food low in calories",\
 "meat", "brown meat", "chicken", "beef", "vegetable pattie", "vegetables",\
 "potato", "starchy foods", "hamburger salad", "salad hamburger"]

sentence_list = [f"The man ate the {food}." for food in food_list]

sentiment_results = classifier(sentence_list)

print(f"The sentiment probability is: {sentiment_results}")

for i, sentence in enumerate(sentence_list):
    print(f"The sentence: \"{sentence}\" is {sentiment_results[i]['label']}\
 with a score of {sentiment_results[i]['score']:.4f}.")