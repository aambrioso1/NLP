"""
This program has two parts:
    
(1) We define a function that calculates the Levenshtein Distance recursively. The function lev(word1, word2) computute the distance between word1 and word2.  The algorithm is based on the mathematical definition of the LD found on the Wikipedia page:  https://en.wikipedia.org/wiki/Levenshtein_distance.  This is not an efficient algorithm since the recursion recalculates many distances over and over again.   But it serves as a good exercise in recursion and as a way to better understand the definition.   The Wikipedia article has information on more efficient algorithms. 

(2) Some data analysis using the LD and short list of words and a panda dataframe.

I found these videos helpful in getting a better understanding of LD:
    https://youtu.be/MiqoA-yF-0M (Benyam Ephrem)
    https://youtu.be/b6AGUjqIPsA (Vivekanand Khyad)
    

    
"""
def lev(word1, word2):
    # Generates the Levenshtein distance between word1 and word2.
    len1 = len(word1)
    len2 = len(word2)
    indicator = 0
    
    # The base case
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    # The recursion
    if len1 and len2 and word1[0] != word2[0]:
        indicator = 1
    return min(
    lev(word1[1:], word2)+1, lev(word1, word2[1:])+1, lev(word1[1:], word2[1:]) + indicator
    )       
 
"""
We create a data frame showing all the distance between the words in wordlist.
"""    
import pandas as pd
wordlist = ['alex', 'anthony', 'sonya', 'erika', 'joseph']
lev_data = []
for i in namelist:
    list1 = []
    for j in namelist:
        list1.append(lev(i,j))
    lev_data.append(list1)
data = dict(zip(namelist,lev_data))

"""
We add a column to the dataframe that shows the total distance of each word in wordlist from all the other words.   This total is a measure of which words are closer to the others (central tendency) and how far apart the words are (variation).
"""
df = pd.DataFrame(data,index = wordlist, columns = wordlist)
print(df)
df['total'] = df.sum()
print(df)


