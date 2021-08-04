# probing_sentences.py

"""
A collection of lists of sentences for different categories of my investigation
For now all sentences are structured around the fill-mask task.
"""

masks = ['[MASK]', '<mask>'] # Bert, Roberta
mask = masks[0]


# basic grammar

grammar = [f"The man and woman run.  The boy {mask} too.",\
  f"The man and woman run.  The girls {mask} too.",\
  f"If it {mask} me, I would do things differently.",\
  f"The man and woman ran.  The boys {mask} too.",\
  f"Today the birds sing.  In the past the birds have{mask}.",\
  f"Yesterday Tommy had {mask} on the swing.",\
]

# common facts
facts = [f"The Emancipation Proclamation was proclaimed by {mask}.",\
  f"The president of the United States now is {mask}.",\
  f"The name of the current president of the United States is {mask}.",\
]

spelling = [f"The word recieve is misspelled. The correct spelling of it is {mask}.",\
]

# basic arithmetic 
arithmetic = [f"Two plus two equals {mask}.",\
  f"2 plus 3 equals {mask}.",\
  f"2 + 3 = {mask}.",\
  f"three plus five equals {mask}.",\
  f"four plus six equals {mask}.",\
  f"seven plus five equals {mask}.",\
  f"two and three make {mask}.",\
  f"two and three make {mask}.",\
  f"Eleven plus twelve equals {mask}.",\
  f"two times two equals {mask}.",\
  f"3 times 3 equals {mask}.",\
  f"The product of two and six equals{mask}.",\
  f"Blicks of 5 is 6. Blicks of 6 is 7. Blicks of 7 is {mask}.",\
  f"Blicks of 5 is 7. Blicks of 7 is 9. Blicks of 9 is {mask}.",\
  f"Blicks of 5 is 10. Blicks of 10 is 15. Blicks of 15 is {mask}.",\
  f"Blicks of 5 is 8. Blicks of 8 is 11. Blicks of 12 is {mask}.",\
  f"Alex is older than Sonya.  Sonya is older than Joseph.  Alex older than {mask}.",\
  f"Alex is older than Sonya.  Sonya is older than Joseph.  Joseph is {mask} than Alex."  
]

# logic, inference
logic = [f"Tom's horse belongs to {mask}.",\
]

color_facts = [f"Mixing the colors red and yellow makes the color {mask}.",\
  f"The color of the raincoat is {mask}.",\
  f"While it rained the man used a black umbrella and wore a {mask} raincoat.",\
]

novel_language = [f"Today the boy wugs.   Yesterday, he {mask}."

]