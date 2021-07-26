# transformers.tutorial_ch2.py

# Chapter 2 of online course at Hugging Face

# Behind the pipeline:  https://huggingface.co/course/chapter2/2?fw=pt

"""
from transformers import AutoTokenizer

# This checkpoint is the default for sentiment-analysis
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Creates tokens that model can process


raw_inputs = ["I really like working in natural language processing.",\
			"I am tired of teaching mathematics.",\
			"The man ate the salad.",\
			"The man ate the hamburger.",\
			"The man ate the cake.",\
			"The man ate the insect.",\
			"The man ate the dog."]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.logits.shape)

print(outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)
"""

# Models:  https://huggingface.co/course/chapter2/3?fw=pt

# Initialize a Bert model

from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config) # This will create a random model

print(f'The BertModel config:\n{config}')


"""
The BertModel config:
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
"""

# To avoid the effort of training a model, we can use a pre-trained model

model = BertModel.from_pretrained("bert-base-cased")

# It is convenient to use Automodel since then the code will be checkpoint-agnostic.
# The model will be pre-trained for the appropriate task.

"""
pre-trained models are cached in the cache folder: ~/.cache/huggingface/transformers
This folder can be customized by setting the HF_HOME environment variable.

List of Bert checkpoints:  https://huggingface.co/models?filter=bert
"""

# Saving a model

dir = "saved_models"
model.save_pretrained(dir)

sequences = [
  "Hello!",
  "Cool.",
  "Nice!",
  "Awesome!",
  "Yikes!"
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# encoded_input = tokenizer(text, return_tensors='pt')
encoded_sequences = [
  [ 101, 7592,  999,  102],
  [ 101, 4658, 1012,  102],
  [ 101, 3835,  999,  102]
]

import torch

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)


"""
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
"""

# Tokenizers: https://huggingface.co/course/chapter2/4?fw=pt 

tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

# ['Jim', 'Henson', 'was', 'a', 'puppeteer']

"""
Two possible ways to tokenize text: (1)  word-base (2) character-based 

Another way
Subword tokenization algorithms rely on the principle that frequently 
used words should not be split into smaller subwords, but rare words 
should be decomposed into meaningful subwords.

Other possibilites:
* byte level BPE - GPT-2
* WordPiece - BERT
* SentencePiece or Unigram - several multilingual models

"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token_dict = tokenizer("Using a Transformer network is simple")

print(f"{token_dict=}")

tokenizer.save_pretrained(dir)

# Encoding:  On this page - https://huggingface.co/course/chapter2/4?fw=pt

"""
Encoding is a two-step process:
(1)  tokenization
(2)  convert to input IDs
"""

# Tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence_1 = "Using a Transformer network is simple"
sequence_2 = "Pretests are useful when one lovingly is performing tokenization"
tokens_1 = tokenizer.tokenize(sequence_1)
tokens_2 = tokenizer.tokenize(sequence_2)

n = 25
print(n * '*') # **********
print(f"{tokens_1=}")
print(f"{tokens_2=}")
print(n * '*') # **********

ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)

print(f"{ids_1=}\n{ids_2=}")
print(n * '*') # **********

encoded_input = tokenizer(sequence_1, return_tensors='pt')
print(n * '*') # **********
print(encoded_input)

print(n * '*') # **********
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

def decode(list):
  print(n * '*') # **********
  print(tokenizer.decode(list))
  print(n * '*') # **********

decode([  101,  7993,   170, 13809, 23763,  2443,  1110,  3014,   102])
decode([100, 1000, 10000])

