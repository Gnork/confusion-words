from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import os
import gensim
from datetime import datetime
import helper

home_dir = os.path.expanduser('~')

### BEGING SETTINGS ###

# text corpus 
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')

# word2vec embeddings
embeddings_file = os.path.join(home_dir, 'embeddings.bin')

# work dir will contain lstm weights in HDF5 format and pickled list of training errors
work_dir = os.path.join(home_dir, 'training_lstm_classification')

# if training should be continued from existing weights, timestamp and start_epoch must be given
# every training is identified by a generated timestamp
# else set values to None
timestamp = None        # string
start_epoch = None      # int

# number of neurons in hidden layer of lstm
hidden_layer_size = 1024

# number of training epochs 
# complete corpus will be given to lstm for training once per epoch 
max_epochs = 100

# after training lstm language model will be applied to this confusion set
# order matters: algorithm will generate rules for occurrences of first word in list
confusion_set = ['than', 'then']

### END SETTINGS ###

# init

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
    
with open(corpus_file) as f:
    sents = [[twp.split('|')[0].lower() for twp in line.split()] for line in f]

# train word2vec model

print()
print('load word2vec embeddings...')
print()	
embeddings = gensim.models.Word2Vec.load(embeddings_file)
word_vector_size = len(embeddings['the'])
print('word vector size: ', word_vector_size)
print()

# extract samples for certain confusion_set from corpus

print('extract confusion set samples from corpus...')
print()
samples = helper.split_samples(helper.all_samples(sents, confusion_set))
print()

# keras lstm model definition

model = Sequential()
model.add(Dropout(0.1))
model.add(LSTM(word_vector_size, hidden_layer_size, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(hidden_layer_size, 1))
model.add(Activation('sigmoid'))

# compile model

print('compile model...')
print()
model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              class_mode="binary")

# load previous states or continue with random initialization
			  
if timestamp and start_epoch:
    errors = helper.load_errors('%s-%d.errors' % (timestamp, start_epoch), work_dir)
    model.load_weights(os.path.join(work_dir, '%s-%d.weights' % (timestamp, start_epoch)))
    print('init previous states...')
    print('timestamp: ', timestamp)
    print('start_epoch: ', start_epoch)
else:
    errors = []
    start_epoch = 0
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print('init new states...')
    print('timestamp: ', timestamp)
print()

# start training

print('start training...')
print()
 
for e in range(max_epochs):
	if e < start_epoch:
	    continue
	    
	error = 0
	counter = 0
	
	for X, y in helper.sample_generator(samples['train'], embeddings):
	    cost = model.train_on_batch(X, y)
	    error += np.mean(cost)
	    counter += 1
	    
	error /= counter
	errors.append(error)
	print('epoch: %d\terror: %f' %((e+1), error))
	print()
	        
	model.save_weights(os.path.join(work_dir, '%s-%d.weights' % (timestamp, (e+1))), overwrite=True)
	helper.save_errors(errors, '%s-%d.errors' % (timestamp, (e+1)), work_dir)
	print('weights saved: %s-%d.weights' % (timestamp, (e+1)))
	print('errors saved: %s-%d.errors' % (timestamp, (e+1)))
	print()

print('end training')
print()
# apply classification

print('apply lstm classification to confusion set...')
print()

train_pred = helper.predictions(model, samples['train'], embeddings)
cv_pred = helper.predictions(model, samples['cv'], embeddings)
test_pred = helper.predictions(model, samples['test'], embeddings)

# find best threshold and print results

helper.classification_results(embeddings, confusion_set, train_pred, cv_pred, test_pred, threshold=None)