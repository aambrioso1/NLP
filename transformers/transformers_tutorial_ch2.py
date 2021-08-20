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


# Models:  https://huggingface.co/course/chapter2/3?fw=pt

# Initialize a Bert model


from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config) # This will create a random model

print(f'The BertModel config:\n{config}')



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


# To avoid the effort of training a model, we can use a pre-trained model

model = BertModel.from_pretrained("bert-base-cased")

# It is convenient to use Automodel since then the code will be checkpoint-agnostic.
# The model will be pre-trained for the appropriate task.


pre-trained models are cached in the cache folder: ~/.cache/huggingface/transformers
This folder can be customized by setting the HF_HOME environment variable.

List of Bert checkpoints:  https://huggingface.co/models?filter=bert


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



from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)


# Tokenizers: https://huggingface.co/course/chapter2/4?fw=pt 

tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

# ['Jim', 'Henson', 'was', 'a', 'puppeteer']


Two possible ways to tokenize text: (1)  word-base (2) character-based 

Another way
Subword tokenization algorithms rely on the principle that frequently 
used words should not be split into smaller subwords, but rare words 
should be decomposed into meaningful subwords.

Other possibilites:
* byte level BPE - GPT-2
* WordPiece - BERT
* SentencePiece or Unigram - several multilingual models



tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
token_dict = tokenizer("Using a Transformer network is simple")

print(f"{token_dict=}")

tokenizer.save_pretrained(dir)

# Encoding:  On this page - https://huggingface.co/course/chapter2/4?fw=pt


Encoding is a two-step process:
(1)  tokenization
(2)  convert to input IDs


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
"""
"""
# Section 2.5 Handling Multiple Sequences

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# Note that the torch.tensor argument below requires a list of ids (extra dimension).
input_ids = torch.tensor([ids]) 
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

print(50*'*')
batched_ids = [ids, ids]
input_ids = torch.tensor(batched_ids) 
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
"""
"""
# Section:  Padding Inputs

batched_ids = [
  [200, 200, 200],
  [200, 200]
]

padding_id = 100

batched_ids = [
  [200, 200, 200],
  [200, 200, padding_id]
]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

# Attention masks

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
]

attention_mask = [
  [1, 1, 1],
  [1, 1, 0]
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)



s1 = "I've been waiting for a HuggingFace course my whole life."
s2 = "I hate this so much!"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


tokens1 = tokenizer.tokenize(s1)
tokens2 = tokenizer.tokenize(s2)
ids1 = tokenizer.convert_tokens_to_ids(tokens1)
ids2 = tokenizer.convert_tokens_to_ids(tokens2)

input_ids1 = torch.tensor([ids1])
input_ids2 = torch.tensor([ids2])
print("Input IDs1:", input_ids1)
print("Input IDs2:", input_ids2)

Input IDs1: tensor([[ 1045,  1521,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012]])
Input IDs2: tensor([[1045, 5223, 2023, 2061, 2172, 999]])



output1 = model(input_ids1)
output2 = model(input_ids2)
print("Logits1:", output1.logits)
print("Logits2:", output2.logits)


batched_ids = [
  [ 101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102],
  [ 101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0, 0, 0, 0, 0, 0,0]
]

# print(batched_ids, len(batched_ids[0])==len(batched_ids[1]))
m1 = [1 for i in range(16)]
m2 = [1 for i in range(8)] + [0 for i in range(8)]
attention_mask = [m1, m2]
print(batched_ids, attention_mask)

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

# Note there is a typo in Section 2 for the attention mask associated with
# "I hate this so much"

# Longer Sequences

""" 
"""
Two models for longer sequences

https://huggingface.co/transformers/model_doc/longformer.html
https://huggingface.co/transformers/model_doc/led.html

There are others.


# For truncating sequences specify the max_sequence_length parameter

sequence = sequence[:max_sequence_length]
"""

# Start here: https://huggingface.co/course/chapter2/6?fw=pt

# Putting it all together

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# We can input a list of sequence, pad, truncate, control the framework.

# "pt" returns PyTorch tensors 
# "tf" returns TensorFlow tensors
# "np" returns NumPy arrays

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

print(f"Logits: {output.logits}")
print(f"Labels: {model.config.id2label}")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
print(f"Result:  {result}")

