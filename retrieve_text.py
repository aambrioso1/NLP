import requests
import re
from collections import Counter
# import speech as sp  #  This a module in Pythonista


text1 = 'http://www.gutenberg.org/cache/epub/5/pg5.txt' # The Constitution
text2 = 'http://www.gutenberg.org/cache/epub/1112/pg1112.txt' # Romeo and Juliet
text3 = 'https://www.gutenberg.org/files/158/158-0.txt'

r = requests.get(text3)
txt = r.text

def words(text): return re.findall(r'\w+', text.lower())

word_list = words(txt)

WORDS = Counter(word_list)
# print(WORDS)

  
print(txt[:5000])

#  sp.say(txt[num2:num2 + 395], 'en-GB')