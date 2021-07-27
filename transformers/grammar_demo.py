from transformers import pipeline

SEQ_NUM = 5 # number of response per task
NUM = 30 # number of asterisks to print

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
  "The word recieve is misspelled. The correct spelling of this word is <mask>."\
  "The president of the United States in 2010 was <mask>."
  "The sum of two and five is <mask>."
  ]

unmasker = pipeline("fill-mask")
for text in sentences:
    response = unmasker(text, top_k=SEQ_NUM)
    print(NUM * '*')
    for i, sentence in enumerate(response):
        print(f"{response[i]['sequence']=}")


for i in range(SEQ_NUM):
    print(f"{response[i]['sequence']}")