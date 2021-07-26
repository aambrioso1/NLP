# On What Transformers Can Do
 
### What is a transformer?


#### From Wikipedia
A [transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) is a deep learning model that adopts the mechanism of attention, differentially weighing the significance of each part of the input data.
This led to the development of pretrained systems such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which were trained with large language datasets, such as Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks. 

[BERT](https://arxiv.org/abs/1810.04805) is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
[Attention is All You Need](https://arxiv.org/abs/1706.03762)

The usual route with created language models that can do natural language tasks is to:
- think of a reasonable language task,
- create a huge database,
- use the database to fine-tune the model for solving the task,
- create an improved model by by comparing new models with the previous one.


### Language Models are Few Shot Learners
I was inspired to write this essay/paper by the article Language Models are Few-Shot Learners (July 2020;  https://arxiv.org/abs/2005.14165).  

What the paper shows is that scaling up the a language model can greatly improve performance.   These larger model can compete with SOTA fine-tuned models on some NLP tasks WITHOUT fine-tuning.   These large scale models can also learned new tasks with a small number of examples or just text interaction.   This last observation is one I would like to explore.   Here is an example from the article on few shot learners:


As you will see from my demonstration, even small models can responds to text communication and demonstate knowledge beyond what they are fine-tune for.   Depending the task the result might be expected, like knowing a history fact, or it might be surprising like being able to do simple arithmetic. 

For now, as I have not learned how fine-tuning a model, I will only use the standard models not above along with the text input associated with the model.

### My Goals
I have been working recently with transformers using the API available at [Hugging Face](https://huggingface.co/).  I have noticed that transformers can respond to text interaction.   I want to imagine that a teacher is probing the general knowlege of a group of students with short text prompts.  The teacher as a questions and a student in the group gives an answer.  But the teachers questions will be in text form and the role of student will be played by a fine-tuned language model.  


Use this question answer approach, I will explore what smaller scale models from the Hugging Face Library can do apart from their pre-tuned tasks and whether models pre-tuned to different tasks behave differently.   

In simpler, but imprecise language, I explore a variety of tasks to see what a transformer, fine-tuned for one of the usual tasks, "knows".   By "know" here I mean that the transformer's response to a task is the same as student who knows the appropriate response would give.  Assume we are using a model trained for fill-mask task.  How will the model fill in the following?

The Emancipation Proclamation was proclaimed by <mask>. (History fact)
Adding five and four results in <mask>. (Simple arithmetic)
The boy likes to run.   Yesterday he <mask>. (basic grammar)
The president of the United States in 2010/2018/now is <mask> (History fact, current events)
The word receive is misspelled.   The correct spelling of this word is <mask>. (spelling)

It is interested to note that the probing here is accomplished in several of the examples by using an initial sentence to prompt the response.

I would like to explore this sort of general knowledge with a good, representative subset of models fined-tuned for various tasks.   For now I will use default models provided through the pipeline method, [transformers.pipeline(task_string)](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline), which accepts the following tasks:

* "feature-extraction"
* "text-classification"
* "sentiment-analysis"
* "token-classification" or "ner"
* "question-answering"
* "fill-mask"
* "summarization"
* "translation_xx_to_yy"
* "text-generation"
* "zero-shot-classification"
* "conversational"

Note that my goal is not to demonstrate the pre-tuned language tasks but rather to explore the general knowledge of the fine-tuned models and their ability to "understand" prompts.

Ultimately my goal to get a better handle on:
* what transformers "know" and don't "know".
* Why they "know" or don't "know" these things.   
* What might be possible for them to "know" without significant changes to the architecture except unrestricting scaling and better datasets.  
