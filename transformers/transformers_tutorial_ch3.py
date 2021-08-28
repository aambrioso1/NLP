# transformers.tutorial_ch3.py

# Chapter 3 of online course at Hugging Face

"""
Things learned in Chapter 3:
* How to prepare a large dataset from the Hub
* How to use the high-level Trainer API to fine-tune a model
* How to use a custom training loop
* How to leverage the 🤗 Accelerate library to easily run that custom training loop on any distributed setup


We are going to train a model on MRPC (Microsoft Research Paraphrase Corpus).
It consists of 5801 pairs of sentence with a label indicating if they are paraphrases
or not.   This database is part of GLUE.

README for datasets: https://github.com/huggingface/datasets/blob/master/README.md

To install datasets using pip:
pip install datasets

"""

from datasets import load_dataset


raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]
raw_validation_dataset = raw_datasets["validation"]

"""
print(50*'*')
print(raw_datasets)
print(50*'*')
print(raw_train_dataset[0])
print(50*'*')
print(f"Features:\n{raw_train_dataset.features}")
"""
# print(50*'*')
# print(raw_train_dataset[15])
"""
print(50*'*')
print(raw_validation_dataset[87])

"""

# Start with preproccessing a dataset.   Watch video first.
# On this page:  https://huggingface.co/course/chapter3/2?fw=pt

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

s1 = raw_train_dataset[15]["sentence1"]
s2 = raw_train_dataset[15]["sentence2"]
# print(f'{s1=} and {s2=}')

# inputs = tokenizer("This is the first sentence.", "This is the second one.")
# print(f"{inputs=}")

input1 = tokenizer(s1)
input2 = tokenizer(s2)
print(f'{s1=} {s1=}')
print(f'{input1=}')
print(f'{input2=}')

inputs2 = tokenizer(s1, s2)
print(50*'*')
print(f"{inputs2=}")
print(50*'*')
print(f"{tokenizer.convert_ids_to_tokens(inputs2['input_ids'])=}")


"""
Start on this page:  https://huggingface.co/course/chapter3/2?fw=pt

Start just after 'Try it Out.' 

"""