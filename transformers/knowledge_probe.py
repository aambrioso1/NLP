	from transformers import pipeline
	import my_sentences

	SEQ_NUM = 5 # number of responses per task
	NUM = 30 # number of asterisks to print for section separator


    models = ["bert-base-uncased", "bert-large-uncased", roberta-base", "roberta-large"] 
	
    # for model in models: To be added after testing.
	unmasker = pipeline("fill-mask", model=models[0])
	for i, text in enumerate(arithmetic, 1):
	    response = unmasker(text, top_k=SEQ_NUM)
	    print(NUM * '*')
	    print(f"[{i}] {text}")
	    for i, sentence in enumerate(response, 1):
	        print(f"* ({i}) {response[i-1]['sequence']}")
