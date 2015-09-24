__author__ = 'Christoph Jansen, HTW Berlin'

import numpy as np
import pickle
import theano
import theano.tensor as T
import os
from collections import Counter

def acs(sentences, preserve_tokens, test_token=None, cv_token=None, test_size=.2, cv_size=.2):
    counters = {}
    for sentence in sentences:
        for token in sentence:
            if token in preserve_tokens:
                counters[token] = counters.get(token, 0) + 1
    sizes = {}
    test_counters = {}
    test_sents = set()
    cv_counters = {}
    cv_sents = set()
    for i, sentence in enumerate(sentences):
        for j, token in enumerate(sentence):
            if counters.get(token, 0) * test_size > test_counters.get(token, 0):
                test_counters[token] = test_counters.get(token, 0) + 1
                test_sents.update([i])
            elif (not i in test_sents) and (counters.get(token, 0) * cv_size > cv_counters.get(token, 0)):
                cv_counters[token] = cv_counters.get(token, 0) + 1
                cv_sents.update([i])
    if test_token:
        for i, sentence in enumerate(sentences):
            if not i in test_sents:
                continue
            for j, token in enumerate(sentence):
                if token == test_token:
                    yield (sentence, j)
    elif cv_token:
        for i, sentence in enumerate(sentences):
            if not i in cv_sents:
                continue
            for j, token in enumerate(sentence):
                if token == cv_token:
                    yield (sentence, j)
    else:
        for i, sentence in enumerate(sentences):
            if (i in test_sents) or (i in cv_sents):
                continue
            yield sentence
            
def token_sequence(sentence, token_embeddings):
    tokens = ['S']+sentence+['E']
    data = []
    targets = []
    for i in range(len(tokens)-1):
        data.append(token_embeddings.token_to_vec(tokens[i]))
        targets.append(token_embeddings.token_to_vec(tokens[i+1]))
    return (theano_cast(data), theano_cast(targets))
    
def vec_to_index(one_hot):
    for i, val in enumerate(one_hot):
        if val > .99:
            return i
    raise Exception('No val greater than .99')

def token_sequence_generator(sentences, token_embeddings):
    for i, sentence in enumerate(sentences):
        yield token_sequence(sentence, token_embeddings)
            
def theano_cast(M):
    return np.asarray(M, dtype=theano.config.floatX)

def save_states(objects, file_name, weights_dir):
    p = os.path.join(weights_dir, file_name)
    with open(p, 'wb') as f:
        for obj in objects:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_name
            
def load_states(file_name, weights_dir, num_objects=17):
    p = os.path.join(weights_dir, file_name)
    loaded_objects = []
    with open(p, 'rb') as f:
        for i in range(num_objects):
            loaded_objects.append(pickle.load(f))
    return loaded_objects

def save_errors(errors, file_name, weights_dir):
    with open(os.path.join(weights_dir, file_name), 'wb') as f:
        pickle.dump(errors, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_name

def load_errors(file_name, weights_dir, num_objects=17):
    p = os.path.join(weights_dir, file_name)
    with open(p, 'rb') as f:
        errors = pickle.load(f)
    return errors
    
def init_weights(shape):
    return theano.shared(theano_cast(np.random.randn(shape[0], shape[1]) * 0.01))

def normalization(token):
    return ''.join(['D' if char.isdigit() else char for char in token]) 

def init_zero_vec(size, one_hot_index=None):
    vec = np.zeros(size, dtype=theano.config.floatX)
    if one_hot_index:
        vec[one_hot_index] = 1
    return theano.shared(vec)
    
def t_softmax(vals, t=1.0):
    exps = [np.exp(val/t) for val in vals]
    s = sum(exps)
    return [val/s for val in exps]
    
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
        
        print('num tokens: ', self.num_tokens)
    
    def token_to_vec(self, token):
        vec = np.zeros(self.num_tokens, theano.config.floatX)
        vec[self.token_to_index.get(token, self.token_to_index['U'])] = 1
        return vec