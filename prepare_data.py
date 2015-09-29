import nltk
import os
import gensim
from collections import Counter

# dense word vector size for word2vec embeddings model
word_vector_size = 128

# dense word vector size for normalized word2vec embeddings model
normalized_word_vector_size = 128

# minimum occurrences of word in corpus for normalized embeddings
# rare words are substituted by character 'U'
min_occurrence = 20

home_dir = os.path.expanduser('~')
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')
embeddings_file = os.path.join(home_dir, 'embeddings.bin')
normalized_embeddings_file = os.path.join(home_dir, 'normalized_embeddings.bin')

nltk.download('brown')

print()
print('format brown corpus and save as file...')

with open(corpus_file, 'w') as f:
	for sentence in nltk.corpus.brown.tagged_sents():
		print(' '.join(['|'.join(tagged_word) for tagged_word in sentence]), file=f)

print()
print('generate word2vec embeddings model and save as file...')

with open(corpus_file) as f:
    sents = [[twp.split('|')[0].lower() for twp in line.split()] for line in f]
	
embeddings = gensim.models.Word2Vec(sents, size=word_vector_size, min_count=1)
embeddings.save(embeddings_file)
	
print()
print('generate normalized word2vec embeddings model and save as file...')

def normalization(token):
    return ''.join(['D' if char.isdigit() else char for char in token.lower()])
	
def count_tokens(sents, min_occurrence):
    c = Counter()
    for sentence in sents:
        c.update(['S'] + sentence + ['E'])
    
    result = Counter()
    for key, val in c.items():
        if val < min_occurrence:
            result.update({'U': val})
        else:
            result.update({key: val})
    return result

class TokenEmbeddings:
    def __init__(self, sents, min_occurrence):
        c = count_tokens(sents, min_occurrence)
        sortable = list(c.keys())
        sortable.sort()
        self.token_to_index = {}
        self.index_to_token = {}
        for i, token in enumerate(sortable):
            self.token_to_index[token] = i
            self.index_to_token[i] = token
        self.num_tokens = i + 1

with open(corpus_file) as f:
    sents = [['S'] + [normalization(twp.split('|')[0]) for twp in line.split()] + ['E'] for line in f]
	
token_embeddings = TokenEmbeddings(sents, min_occurrence)

sents = [[token if token in token_embeddings.token_to_index else 'U' for token in s] for s in sents]
	
embeddings = gensim.models.Word2Vec(sents, size=word_vector_size, min_count=1)
embeddings.save(normalized_embeddings_file)

print()
print('done')