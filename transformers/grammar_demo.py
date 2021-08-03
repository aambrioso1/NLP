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
  "Today the birds sing.  In the past the birds have <mask>.",\
  "Yesterday Tommy had <mask> on the swing.",\
  "The word recieve is misspelled. The correct spelling of it is <mask>.",\
  "The president of the United States now is <mask>.",\
  "The name of the current president of the United States is <mask>.",\
  "Two plus two equals <mask>.",\
  "Three plus three equals <mask>.",\
  "Two plus five equals <mask>.",\
  "Two plus three equals <mask>.",\
  "Three plus six equals <mask>.",\
  "Three and six make <mask>.",\
  "Eleven plus twelve equals <mask>.",\
  "Tom's horse belongs to <mask>.",\
  "Mixing the colors red and yellow makes the color <mask>.",\
  "The color of the raincoat is <mask>.",\
  "The usual color of a raincoat is <mask>."\
  ]

unmasker = pipeline("fill-mask")
for i, text in enumerate(sentences, 1):
    response = unmasker(text, top_k=SEQ_NUM)
    print(NUM * '*')
    print(f"[{i}] {text}")
    for i, sentence in enumerate(response, 1):
        print(f"({i}) {response[i-1]['sequence']}")


        
"""
Output:  fine-tuned task: fill-mask
Knowledge tested
[1]  Grammar: subject/verb agreement
[2]  Grammar: subject/verb agreement
[5]  History fact
[8]  Spelling
[11] - [17]  Simple arthmetic
[18] Content reflection/logic
[19], [20]  Color mixing facts
******************************
[1] The man and woman run.  The boy <mask> too.
(1) The man and woman run.  The boy runs too.
(2) The man and woman run.  The boy cries too.
(3) The man and woman run.  The boy run too.
(4) The man and woman run.  The boy goes too.
(5) The man and woman run.  The boy dies too.
******************************
[2] The man and woman run.  The girls <mask> too.
(1) The man and woman run.  The girls run too.
(2) The man and woman run.  The girls go too.
(3) The man and woman run.  The girls scatter too.
(4) The man and woman run.  The girls win too.
(5) The man and woman run.  The girls do too.
******************************
[3] If it <mask> me, I would do things differently.
(1) If it suited me, I would do things differently.
(2) If it were me, I would do things differently.
(3) If it was me, I would do things differently.
(4) If it bothered me, I would do things differently.
(5) If it suits me, I would do things differently.
******************************
[4] The man and woman ran.  The boys <mask> too.
(1) The man and woman ran.  The boys ran too.
(2) The man and woman ran.  The boys did too.
(3) The man and woman ran.  The boys bolted too.
(4) The man and woman ran.  The boys fled too.
(5) The man and woman ran.  The boys run too.
******************************
[5] The Emancipation Proclamation was proclaimed by <mask>.
(1) The Emancipation Proclamation was proclaimed by Congress.
(2) The Emancipation Proclamation was proclaimed by Lincoln.
(3) The Emancipation Proclamation was proclaimed by Jefferson.
(4) The Emancipation Proclamation was proclaimed by Napoleon.
(5) The Emancipation Proclamation was proclaimed by FDR.
******************************
[6] Today the birds sing.  In the past the birds have <mask>.
(1) Today the birds sing.  In the past the birds have sung.
(2) Today the birds sing.  In the past the birds have sang.
(3) Today the birds sing.  In the past the birds have flown.
(4) Today the birds sing.  In the past the birds have died.
(5) Today the birds sing.  In the past the birds have danced.
******************************
[7] Yesterday Tommy had <mask> on the swing.
(1) Yesterday Tommy had resumed on the swing.
(2) Yesterday Tommy had surgery on the swing.
(3) Yesterday Tommy had been on the swing.
(4) Yesterday Tommy had returned on the swing.
(5) Yesterday Tommy had fallen on the swing.
******************************
[8] The word recieve is misspelled. The correct spelling of it is <mask>.
(1) The word recieve is misspelled. The correct spelling of it is incorrect.
(2) The word recieve is misspelled. The correct spelling of it is corrected.
(3) The word recieve is misspelled. The correct spelling of it is unknown.
(4) The word recieve is misspelled. The correct spelling of it is correct.
(5) The word recieve is misspelled. The correct spelling of it is omitted.
******************************
[9] The president of the United States now is <mask>.
(1) The president of the United States now is dead.
(2) The president of the United States now is gone.
(3) The president of the United States now is president.
(4) The president of the United States now is retiring.
(5) The president of the United States now is illegitimate.
******************************
[10] The name of the current president of the United States is <mask>.
(1) The name of the current president of the United States is unknown.
(2) The name of the current president of the United States is incorrect.
(3) The name of the current president of the United States is withheld.
(4) The name of the current president of the United States is unclear.
(5) The name of the current president of the United States is uncertain.
******************************
[11] Two plus two equals <mask>.
(1) Two plus two equals zero.
(2) Two plus two equals three.
(3) Two plus two equals four.
(4) Two plus two equals two.
(5) Two plus two equals one.
******************************
[12] Three plus three equals <mask>.
(1) Three plus three equals zero.
(2) Three plus three equals four.
(3) Three plus three equals three.
(4) Three plus three equals two.
(5) Three plus three equals five.
******************************
[13] Two plus five equals <mask>.
(1) Two plus five equals zero.
(2) Two plus five equals four.
(3) Two plus five equals seven.
(4) Two plus five equals three.
(5) Two plus five equals six.
******************************
[14] Two plus three equals <mask>.
(1) Two plus three equals zero.
(2) Two plus three equals four.
(3) Two plus three equals three.
(4) Two plus three equals two.
(5) Two plus three equals five.
******************************
[15] Three plus six equals <mask>.
(1) Three plus six equals zero.
(2) Three plus six equals seven.
(3) Three plus six equals nine.
(4) Three plus six equals four.
(5) Three plus six equals eight.
******************************
[16] Three and six make <mask>.
(1) Three and six make plays.
(2) Three and six make contact.
(3) Three and six make outs.
(4) Three and six make appearances.
(5) Three and six make mistakes.
******************************
[17] Eleven plus twelve equals <mask>.
(1) Eleven plus twelve equals thirteen.
(2) Eleven plus twelve equals twelve.
(3) Eleven plus twelve equals eleven.
(4) Eleven plus twelve equals fourteen.
(5) Eleven plus twelve equals sixteen.
******************************
[18] Tom's horse belongs to <mask>.
(1) Tom's horse belongs to me.
(2) Tom's horse belongs to Tom.
(3) Tom's horse belongs to him.
(4) Tom's horse belongs to us.
(5) Tom's horse belongs to Daddy.
******************************
[19] Mixing the colors red and yellow makes the color <mask>.
(1) Mixing the colors red and yellow makes the color brighter.
(2) Mixing the colors red and yellow makes the color lighter.
(3) Mixing the colors red and yellow makes the color darker.
(4) Mixing the colors red and yellow makes the color richer.
(5) Mixing the colors red and yellow makes the color clearer.
******************************
[20] The color of the raincoat is <mask>.
(1) The color of the raincoat is purple.
(2) The color of the raincoat is unknown.
(3) The color of the raincoat is pink.
(4) The color of the raincoat is orange.
(5) The color of the raincoat is muted.
"""
