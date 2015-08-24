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
    
def accuracy(flags):
    c = correct(flags)
    w = wrong(flags)
    if c == 0:
        return 0
    return c / (c + w)

def correct(flags):
    tn, fp, tp, fn = flags
    return tp + tn

def wrong(flags):
    tn, fp, tp, fn = flags
    return fp + fn

def precision(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fp)

def recall(flags):
    tn, fp, tp, fn = flags
    if tp == 0:
        return 0
    return tp / (tp + fn)

def f_score(flags, beta=1):
    p = precision(flags)
    r = recall(flags)
    b = beta
    if p == 0 or r == 0:
        return 0
    return (1 + b*b) * (p*r / (b*b*p + r))

def w_flags(flags, artificial_error):
    e = artificial_error
    tn, fp, tp, fn = flags
    w_tn = tn * (1.0-e)
    w_fp = fp * (1.0-e)
    w_tp = tp * e
    w_fn = fn * e
    return (w_tn, w_fp, w_tp, w_fn)

def w_error_corpus(flags):
    tn, fp, tp, fn = flags
    error_corpus_weight = (tn + fp) / (tp + fn)
    w_tp = tp * error_corpus_weight
    w_fn = fn * error_corpus_weight
    return (tn, fp, w_tp, w_fn)

def print_stats(flags):
    TN, FP, TP, FN = flags
    print('TN\tFP\tTP\tFN')
    print('%d\t%d\t%d\t%d' % (TN, FP, TP, FN))
    print('accuracy:\t%f' %accuracy(flags))
    print('precision\':\t%f' %precision(w_flags(w_error_corpus(flags), 0.01)))
    print('recall\':\t%f' %recall(w_flags(w_error_corpus(flags), 0.01)))
    print('f_score:\t%f' %f_score(w_flags(w_error_corpus(flags), 0.01)))
    print('f_05_score:\t%f' %f_score(w_flags(w_error_corpus(flags), 0.01), beta=0.5))
    
def predictions(model, samples, word_embeddings):
    data = []
    for X, y in sample_generator(samples, word_embeddings):
        t = (model.predict(X)[0][0], y[0][0])
        data.append(t)
    return data

def lm_results(cv_predictions, test_predictions, cv_0, cv_1, test_0, test_1, embeddings, confusion_set):
    temps = np.arange(0.7, 1.3, 0.1)
    thresholds = np.arange(0, 100, 5)
    
    best_threshold = 0
    best_temp = 0
    best_score = 0
    
    print('find best combination of temperature and threshold...')
    print()
    
    print('temp\tthreshold\tcv_score\ttest_score')
    for i, temp in enumerate(temps):
        for j, threshold in enumerate(thresholds):
            cv_flags = stats(cv_predictions, cv_0, cv_1, embeddings, threshold, temp)
            test_flags = stats(test_predictions, test_0, test_1, embeddings, threshold, temp)
            cv_score = f_score(w_flags(w_error_corpus(cv_flags), 0.01), 0.5)
            test_score = f_score(w_flags(w_error_corpus(test_flags), 0.01), 0.5)
            print('%.1f\t%d\t\t%f\t%f' % (temp, threshold, cv_score, test_score))
            if cv_score > best_score:
                best_score = cv_score
                best_temp = temp
                best_threshold = threshold
    print()
    
    cv_flags = stats(cv_predictions, cv_0, cv_1, embeddings, best_threshold, best_temp)
    test_flags = stats(test_predictions, test_0, test_1, embeddings, best_threshold, best_temp)
    
    print('best temp and threshold:')
    print(best_temp, best_threshold)
    print()
    print('CV:')
    print_stats(cv_flags)
    print()
    print('Test:')
    print_stats(test_flags)
        
def stats(predictions, samples_0, samples_1, embeddings, threshold, temp):
    TN = 0
    FP = 0
    TP = 0
    FN = 0
    
    pred_0_correct, pred_0_wrong, pred_1_correct, pred_1_wrong = predictions
    
    for c0, w0, (sentence, index) in zip(pred_0_correct, pred_0_wrong, samples_0):
        p_c0 = char_sequence_probability(sentence, c0, embeddings, t=temp)
        p_w0 = char_sequence_probability(sentence, w0, embeddings, t=temp)
        if p_w0 > p_c0 + threshold:
            FP += 1
        else:
            TN += 1
            
    for c1, w1, (sentence, index) in zip(pred_1_correct, pred_1_wrong, samples_1):
        p_c1 = char_sequence_probability(sentence, c1, embeddings, t=temp)
        p_w1 = char_sequence_probability(sentence, w1, embeddings, t=temp)
        if p_c1 > p_w1 + threshold:
            TP += 1
        else:
            FN += 1
    
    return (TN, FP, TP, FN)

def char_sequence_probability(sentence, predictions, embeddings, t=1.0):
    probabilities = t_softmax(predictions, t=t)
    log_p = 0
    sequence = list(' '.join(sentence))[1:]
    for c, p in zip(sequence, probabilities):
        index = embeddings[c]
        log_p += np.log(p[index])
    return log_p
    
def t_softmax(vals, t=1.0):
    exps = [np.exp(val/t) for val in vals]
    s = sum(exps)
    return [val/s for val in exps]