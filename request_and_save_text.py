# Requests is an elegant and simple HTTP library for Python, built for human beings.
import requests as req

import os # Standard library.  Operating system interfaces. 


# Use requests to collected text from various websites (mostly Gutenberg.org).
response = req.get('https://www.gutenberg.org/files/70/70-0.txt') # What is Man? And Other Stories
text = str(response.text.encode('utf8').decode('ascii', 'ignore'))

# Write text to the hard drive
cwd = os.getcwd() 
# path = cwd + '\\what_is_man.txt' 
path = 'C:\\Users\\Alex\\OneDrive - Hillsborough Community College\\Sabbatical 2020_21\\Great Books Discussion Group\\what_is_man.txt'
file = open(path,'w')
file.write(text)
file.close()
