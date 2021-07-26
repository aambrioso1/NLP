# On What Transformers Can Do

 
### What is a transformer?

pre-train on a huge language corpus
think of a reasonable language task, create a huge database
Try to improve models and guage the result by comparing previous model performance to the new one.


### Language Models are Few Shot Learners
I was inspired to write this essay/paper by the article Language Models are Few-Shot Learners (July 2020;  https://arxiv.org/abs/2005.14165).  

What the paper shows is that scaling up the a language model can greatly improve performance.   These larger model can be even compete with SOTA fine-tuned models on some NLP tasks.   The bigger models can also learned new tasks with a small number of examples or just text interaction!


Example from article of prompt:





### My Goals
I have been working recently with transformers using the API available at [Hugging Face](https://huggingface.co/).  I have noticed that transformers can respond to text interaction.   I want to imagine that a teacher is probing the general knowlege of a group of students with short text prompts but have fine-tuned model play the role of the student.  I will explore what smaller scale models from the Hugging Face Library can do apart from the pre-tuned tasks and whether models pre-tuned to different tasks behave differently.   I explore a variety for tasks to see what a transformer, fine-tuned for one of the usual tasks, knows.   For example.   Given a model trained for fill mask, how will the model answer the follow?

The Emancipation Proclamation was proclaimed by <mask>.
Adding five and four results in <mask>.
The boy likes to run.   Yesterday he <mask>.
The president of the United States in 2010/2018 is <mask>
The word receive is misspelled.   The correct spelling is <mask>


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

Note that my goal is not to demonstrate the language tasks but rather to explore the general knowledge of the fine-tuned models and their ability to "understand" prompts.

Ultimately my goal to get a better handle on:
* what transformers can and cannot do.
* Why the can or cannot do these things.   
* What might be important for them to do? and 
