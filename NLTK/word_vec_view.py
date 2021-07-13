# Found here:  https://www.kaggle.com/alvations/word2vec-embedding-using-gensim-and-nltk
# Some basic operation with word2vec and gensim.

import gensim
from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

import numpy as np
labels = []
count = 0
max_count = 50
X = np.zeros(shape=(max_count,len(model['university'])))

for term in model.vocab:
    X[count] = model[term]
    labels.append(term)
    count+= 1
    if count >= max_count: break

# It is recommended to use PCA first to reduce to ~50 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_50 = pca.fit_transform(X)

# Using TSNE to further reduce to 2 dimensions
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=0)
Y = model_tsne.fit_transform(X_50)

# Show the scatter plot
import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], 20)

# Add labels
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 10)

plt.show()

"""
See line 48.  Need to work out who to get this file:  
FileNotFoundError: [Errno 2] No such file or directory: 'GoogleNews-vectors-negative300.bin.gz'

import gensim
from gensim.models.word2vec import Word2Vec
# Load the binary model
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True);

# Only output word that appear in the Brown corpus
from nltk.corpus import brown
words = set(brown.words())
print (len(words))

# Output presented word to a temporary file
out_file = 'pruned.word2vec.txt'
f = open(out_file,'wb')

word_presented = words.intersection(model.vocab.keys())
f.write('{} {}\n'.format(len(word_presented),len(model['word'])))

for word in word_presented:
    f.write('{} {}\n'.format(word, ' '.join(str(value) for value in model[word])))

f.close()
"""