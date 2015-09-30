__author__ = 'Christoph Jansen, HTW Berlin'

import numpy as np
import pickle
import theano
import theano.tensor as T
import os

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
def theano_cast(M):
    return np.asarray(M, dtype=theano.config.floatX)

def char_embeddings(sentences):
    chars = {' ', 'S', 'E'}
    for sentence in sentences:
        for token in sentence:
            chars.update(list(token))
    sortable = list(chars)
    sortable.sort()
    char_to_index = {}
    for i, c in enumerate(sortable):
        char_to_index[c] = i
    return char_to_index

def char_sequence(sentence, embeddings):
    chars = ['S']+list(' '.join(sentence))+['E']
    data = []
    targets = []
    for i in range(len(chars)-1):
        data.append(index_to_vec(chars[i], embeddings))
        targets.append(index_to_vec(chars[i+1], embeddings))
    return (theano_cast(data), theano_cast(targets))

def index_to_vec(c, embeddings):
    vec = np.zeros(len(embeddings), dtype=theano.config.floatX)
    vec[embeddings[c]] = 1
    return vec

def index_to_embedded(index, embeddings):
    for key, val in embeddings.items():
        if val == index:
            return key
    raise Exception('Index is not in embeddings')
    
def sampled_chars_to_string(sample, embeddings):
    chars = [index_to_embedded(index, embeddings) for index in sample]
    return ''.join(chars)
    
def vec_to_index(one_hot):
    for i, val in enumerate(one_hot):
        if val > .99:
            return i
    raise Exception('No val greater than .99')

def char_sequence_generator(sentences, embeddings):
    for i, sentence in enumerate(sentences):
        seq = char_sequence(sentence, embeddings)
        yield seq

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

def init_zero_vec(size, one_hot_index=None):
    vec = np.zeros(size, dtype=theano.config.floatX)
    if one_hot_index:
        vec[one_hot_index] = 1
    return theano.shared(vec)
    
def t_softmax(vals, t=1.0):
    exps = [np.exp(val/t) for val in vals]
    s = sum(exps)
    return [val/s for val in exps]