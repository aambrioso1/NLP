# transformers.tutorial_ch3.py

# Chapter 3 of online course at Hugging Face

"""
Things learned in Chapter 3:
* How to prepare a large dataset from the Hub
* How to use the high-level Trainer API to fine-tune a model
* How to use a custom training loop
* How to leverage the 🤗 Accelerate library to easily run that custom training loop on any distributed setup


We are going to train a model on MRPC (Microsoft Research Paraphrase Corpus).
Paper on this data set:  https://aclanthology.org/I05-5002.pdf
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
https://huggingface.co/course/chapter3/2?fw=

BERT is pretrained with token type IDs, and on top of the masked
language modeling objective we talked about in Chapter 1, it has an 
additional objective called next sentence prediction. The goal with
this task is to model the relationship between pairs of sentences.

"""
print(50*"*")
print(50*"*")

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"{tokenized_datasets=}")

"""
Dynamic padding (DataCollatorWithPadding) - rather padding the whole dataset 
at once it is often more efficient to postpone padding and pad batches individually.
The downside is that batches will all have different size.  Dynamic padding is more
useful with GPUs.   TPUs perfer fixed shapes.
"""

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}

print(f'Length of samples: {[len(x) for x in samples["input_ids"]]}')

batch = data_collator(samples)

item_set = {k: v.shape for k, v in batch.items()}
print(f'Batch items: {item_set}')

"""
Start at bottom of:  https://huggingface.co/course/chapter3/2?fw=pt
Try it out! Replicate the preprocessing on the GLUE SST-2 dataset. 
"""

raw_datasets2 = load_dataset("glue", "sst2")
raw_train_dataset2 = raw_datasets2["train"]
raw_validation_dataset2 = raw_datasets2["validation"]
print('\n' + 50*'*')
print('SST-2')
print(50*'*')
print(f'{raw_datasets2=}')
print(50*'*')
print(raw_train_dataset2[0])
print(50*'*')
print(f"Features:\n{raw_train_dataset2.features}")

print(50*'*')
print(raw_train_dataset2[15])

print(50*'*')
print(raw_validation_dataset2[87])

def tokenize_function2(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets2 = raw_datasets2.map(tokenize_function2, batched=True)
print(f'{tokenized_datasets2=}')


