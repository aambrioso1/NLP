"""
A spell check program by Peter Norvig.
Found here:
http://norvig.com/spell-correct.html
"""

import re
from collections import Counter


def words(text): return re.findall(r'\w+', text.lower())

"""
This program uses a file called big.txt.   This is a file containing words.   It is used in two ways:  (1) as a collection of correctly spelled words.  (2)  To compute the relative frequency of words in English.
import requests
"""
"""
# This code uses requests to store big.txt in txt.
r = requests.get('https://norvig.com/big.txt')
txt = r.text
"""

f = open('C:/Users/aambrioso/OneDrive - Hillsborough Community College/Programming/big.txt')
txt = f.read()
f.close()

WORDS = Counter(words(txt))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    # A list of all the ways to split word into two parts. 
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    # A list of all strings formed by deleting one letter from word.
    deletes    = [L + R[1:]               for L, R in splits if R]
    # A list of all strings formed by transposing two consecutive letters in word. 
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    # A list of all strings formed by replacing a letter into word.
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    # A list of all strings formed by inserting a letter into word.
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    
# print(correction('mahtematics'))
    
import itertools as it

def scramble(word):
    lst = []
    for i in range(1, len(word)+1):
        perms1 = it.permutations(word,i)
        for j in perms1:
            perms2 = ''.join(j)
            lst.append(perms2)
    return lst
    

