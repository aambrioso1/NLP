from transformers import pipeline

import probing_sentences as ps

SEQ_NUM = 5 # number of responses per task
NUM = 35 # number of asterisks to print for section separator


models = ["bert-base-cased", "roberta-base"] 
model = models[0]

# for model in models: To be added after testing.
unmasker = pipeline("fill-mask", model=model)

print(f'**********{model}**********\n')
for i, text in enumerate(ps.arithmetic, 1):
	print(text)
	response = unmasker(text, top_k=SEQ_NUM)
	print(NUM * '*')
	print(f"[{i}] {text}")
	for i, sentence in enumerate(response, 1):
		print(f"* ({i}) {response[i-1]['sequence']}")
	print('\n')

