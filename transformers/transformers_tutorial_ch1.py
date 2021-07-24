# transformers-tutorial.py

"""
Code for the HuggingFace tutorial found here:
https://huggingface.co/course

Use this script c:/bin/transformers_venv to start

The Hugging Face model hub is here:

https://huggingface.co/models

Available pipelines found here:

https://huggingface.co/transformers/main_classes/pipelines.html


"""

from transformers import pipeline

SEQ_NUM = 5 # number of response per task
LEN = 100 # number of characters in a response
NUM = 30 # number of asterisks to print

"""
# Sentiment analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))


sentiment_probability = classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
    "I am ambivalent about the effectiveness that course of action."
])

print(f"The sentiment probability is: {sentiment_probability}")

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
probabilities = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(f'The probabilities are {probabilities}')

# text = "In this course, we will teach you how to"




# Text generation with default model

generator = pipeline("text-generation")
response = generator(text, num_return_sequences=SEQ_NUM, max_length=LEN)

# Text generation with chosen model
generator = pipeline("text-generation", model="distilgpt2")
response = generator(text, num_return_sequences=SEQ_NUM, max_length=LEN)

for i in range(SEQ_NUM):
    print(f"({i+1}).  The generated text is the following:\n\
{41 * '*'}\n\
{response[i]['generated_text']}")

# Some experiments in grammar.
# birds sentence: Brown, p. 287
# skaters sentence: Brown, p. 278
sentences = ["The man and woman run.  The boy <mask> too.",\
  "The man and woman run.  The girls <mask> too.",\
  "If it <mask> me, I would do things differently.",\
  "The man and woman ran.  The boys <mask> too.",\
  "The Emancipation Proclamation was proclaimed by <mask>.",\
  "Today the birds sing.  In the past the birds have <mask>.",
  "Yesterday Tommy had <mask> on the swing."\
  ]

unmasker = pipeline("fill-mask")
for text in sentences:
    response = unmasker(text, top_k=SEQ_NUM)
    print(NUM * '*')
    for i, sentence in enumerate(response):
        print(f"{response[i]['sequence']=}")


for i in range(SEQ_NUM):
    print(f"{response[i]['sequence']}")

# Name entity recognition

#ner = pipeline("ner", model="vblagoje/bert-english-uncased-finetuned-pos", grouped_entities=True)
# print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))
"""
# Question answering


question_answerer = pipeline("question-answering")
response = question_answerer(
    question="What is Bert?",
    context="""Bidirectional Encoder Representations\
    from Transformers (BERT) is a transformer-based\
    machine learning technique for natural language\
    processing (NLP) pre-training developed by Google.\
    BERT was created and published in 2018 by Jacob\
    Devlin and his colleagues from Google.[1][2] As \
    of 2019, Google has been leveraging BERT to better \
    understand user searches.[3]\
    The original English-language BERT has two models:\
    [1] (1) the BERTBASE: 12 Encoders with 12 bidirectional \
    self-attention heads, and (2) the BERTLARGE: 24 Encoders \
    with 16 bidirectional self-attention heads. Both models \
    are pre-trained from unlabeled data extracted from the\
    BooksCorpus[4] with 800M words and English Wikipedia\
    with 2,500M words.[5]
    """)

print(f'{response=}')

"""
# Summarization


summarizer = pipeline("summarization")
# response = summarizer("
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.")

# print(f'{response=}')
# Translation

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
response = translator("Ce cours est produit par Hugging Face.")

print(f'{response=}')

translator = pipeline("translation_en_to_es", model="mrm8488/mbart-large-finetuned-opus-en-es-translation")
response = translator("I am enjoying learning with Hugging Face.")
"""

# print(f'{response=}')
