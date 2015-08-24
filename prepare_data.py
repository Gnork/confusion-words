import nltk
import os
import gensim

# dense word vector size for word2vec
word_vector_size = 128

home_dir = os.path.expanduser('~')
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')
embeddings_file = os.path.join(home_dir, 'embeddings.bin')

nltk.download('brown')

print()
print('format brown corpus and save as file...')

with open(corpus_file, 'w') as f:
	for sentence in nltk.corpus.brown.tagged_sents():
		print(' '.join(['|'.join(tagged_word) for tagged_word in sentence]), file=f)

print()
print('generate word2vec model and save as file...')

with open(corpus_file) as f:
    sents = [[twp.split('|')[0].lower() for twp in line.split()] for line in f]
	
embeddings = gensim.models.Word2Vec(sents, size=word_vector_size, min_count=1)
embeddings.save(embeddings_file)