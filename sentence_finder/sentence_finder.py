"""
https://stackoverflow.com/questions/3549075/regex-to-find-all-sentences-of-text

pattern for finding a sentence: Upercase, then anything that is not in (.!?), then one of them

Things to do
Try different words
Try relative occurances (word frequency/total number of words)

"""
import requests as req
import re
import os # Standard library.  Operating system interfaces. 

"""
text1 = This is a short paragraph.   It should test whether my regular expression is working.\
It should find all the sentences in this text.   Then the code lists all the sentences with 'it' in them.\
Does it?  I hope it does!\
"""
"""
# Index for all books in the GB of the numbers used in these URLs:  https://www.gutenberg.org/dirs/GUTINDEX.ALL
r1 = req.get('https://www.gutenberg.org/cache/epub/1777/pg1777.txt') # Romeo and Juliet
r2 = req.get('https://www.gutenberg.org/cache/epub/1787/pg1787.txt') # Hamlet
r3 = req.get('https://www.gutenberg.org/cache/epub/1503/pg1503.txt') # King Richard III
"""

def sentence_finder(word, URL, filename="sentences"):
	""" Find the sentences in the text at "URL" that have an occurence of "word".
	Each sentence is numbered and stored in the CWD in a file called "filename"

	Args:
		word:  A string of a target word.
		URL:  A string of a URL containing a text file (like a etext in the Gutenberg project)
		filename:  A string for a filename to store the sentences.
	Returns:
		The string "{filename} has {count}s sentences with the word {word} in them." 
	Dependencies:
		The os(built-in), re(built-in), and requests(third-party) libraries.
	
	"""
	# Get the information form the URL and turn it into text.
	r2 = req.get(URL) # Hamlet
	text = str(r2.text[1:])
	
	# Create a regular expression to find all the sentences in a string of text.
	pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
	sentences = pat.findall(text)
	
	# Open a file to store the sentences with word in them.
	path = os.getcwd() + '\\' + filename + '_sentence_finder.txt'
	with open(path, 'a') as file:
		# Iterate through the sentences to find the ones that have word in them and add them to the file.
		count = 0 # Counts the number of sentences
		for item in sentences:
			if (word or word.capitalize()) in item:
				count += 1
				new_text = f'({count}).  {item}\n'
				file.write(new_text)
				
	return f'{filename} has {count} sentences with the word "{word}" in them.'

print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1777/pg1777.txt', 'Romeo_and_Juliet'))
print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1787/pg1787.txt', 'Hamlet'))
print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1503/pg1503.txt', 'King_Richard_III'))
print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1135/pg1135.txt', 'The_Tempest'))
print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1514/pg1514.txt', 'A_Midsummer_Night_s_Dream'))
print(sentence_finder('love', 'https://www.gutenberg.org/cache/epub/1534/pg1534.txt', 'Antony and Cleopatra'))