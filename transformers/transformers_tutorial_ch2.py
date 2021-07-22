# transformers.tutorial_ch2.py

from transformers import AutoTokenizer

# This checkpoint is the default for sentiment-analysis
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Creates tokens that model can process


raw_inputs = ["I really like working in natural language processes.",\
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



