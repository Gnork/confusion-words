import numpy as np
import theano
import os
import pickle

def all_samples(sentences, confusion_set):
    
    token_a = confusion_set[0]
    token_b = confusion_set[1]
    
    samples = []
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            if token == token_a:
                clone = sentence[:]
                t = (False, i, clone)
                samples.append(t)
            elif token == token_b:
                clone = sentence[:]
                clone[i] = token_a
                t = (True, i, clone)
                samples.append(t)          
    return samples

def split_samples(samples, test_size=.2, cv_size=.2):
    num_correct = 0
    num_wrong = 0
    
    for sample in samples:
        if sample[0]:
            num_wrong += 1
        else:
            num_correct += 1
            
    print('positive samples:', num_correct)
    print('negative samples:', num_wrong)
            
    test_correct = []
    cv_correct = []
    test_wrong = []
    cv_wrong = []
            
    split = {'train': [],
             'test' : [],
             'cv': []}
    
    for sample in samples:
        if sample[0]:
            if len(test_wrong) < num_wrong * test_size:
                test_wrong.append(sample)
            elif len(cv_wrong) < num_wrong * cv_size:
                cv_wrong.append(sample)
            else:
                split['train'].append(sample)
        else:
            if len(test_correct) < num_correct * test_size:
                test_correct.append(sample)
            elif len(cv_correct) < num_correct * cv_size:
                cv_correct.append(sample)
            else:
                split['train'].append(sample)
                
    split['test'] = test_correct + test_wrong
    split['cv'] = cv_correct + cv_wrong
    return split
        
def sample_generator(samples, word_embeddings):
    vec_size = len(word_embeddings['the'])
    
    for sample in samples:
        is_wrong = sample[0]
        sentence = sample[2]
        sample_size = len(sentence)
        
        X = np.zeros((1, sample_size, vec_size), dtype=theano.config.floatX)
        y = np.zeros((1, 1), dtype=theano.config.floatX)
        
        for i, token in enumerate(sentence):
            embedding = word_embeddings[token]
            for j, val in enumerate(embedding):
                X[0, i, j] = val
        y[0, 0] = is_wrong
        
        yield (X, y)
        
def save_errors(errors, file_name, weights_dir):
    with open(os.path.join(weights_dir, file_name), 'wb') as f:
        pickle.dump(errors, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_name
            
def load_errors(file_name, weights_dir):
    with open(os.path.join(weights_dir, file_name), 'rb') as f:
        errors = pickle.load(f)
    return errors
    
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
    tp = tp * error_corpus_weight
    fn = fn * error_corpus_weight
    return (tn, fp, tp, fn)

def print_stats(flags, error_rate):
    TN, FP, TP, FN = flags
    if error_rate != 0.5:
        flags = w_error_corpus(flags)
    
    print('TN\tFP\tTP\tFN')
    print('%d\t%d\t%d\t%d' % (TN, FP, TP, FN))
    acc = accuracy(flags)
    print('accuracy:\t%f' %acc)
    pre = precision(w_flags(flags, error_rate))
    print('precision\':\t%f' %pre)
    rec = recall(w_flags(flags, error_rate))
    print('recall\':\t%f' %rec)
    print('f_score:\t%f' %f_score(w_flags(flags, error_rate)))
    print('f_05_score:\t%f' %f_score(w_flags(flags, error_rate), beta=0.5))
    
    
def predictions(model, samples, word_embeddings):
    data = []
    for X, y in sample_generator(samples, word_embeddings):
        t = (model.predict(X)[0][0], y[0][0])
        data.append(t)
    return data
        
def stats(predictions, threshold):
    TN = 0
    FP = 0
    TP = 0
    FN = 0
    
    for prediction, y in predictions:
        if y > .5:
            if prediction >= threshold:
                TP += 1
            else:
                FN += 1
        else:
            if prediction >= threshold:
                FP += 1
            else:
                TN += 1
    
    return (TN, FP, TP, FN)

def classification_results(word_embeddings, confusion_set, train_pred, cv_pred, test_pred, annotated_pred=None, error_rate=0.01, threshold=None):
    print('find best threshold...')
    print()
    
    print('threshold\tcv_score\ttest_score')
    
    if not threshold:
        thresholds = []
        for i in range(1, 10):
            diff = 1.0/(10**i)
            thresholds.append(1-diff)
        thresholds.append(1.0)

        best_threshold = 0
        best_f_score = 0

        for i, threshold in enumerate(thresholds):
            cv_flags = stats(cv_pred, threshold)
            cv_score = f_score(w_flags(w_error_corpus(cv_flags), error_rate), beta=0.5)

            test_flags = stats(test_pred, threshold)
            test_score = f_score(w_flags(w_error_corpus(test_flags), error_rate), beta=0.5)

            print('%d\t\t%f\t%f' % (threshold, cv_score, test_score))
            
            if cv_score >= best_f_score:
                best_f_score = cv_score
                best_threshold = threshold
    else:
        best_threshold = threshold
            
    print('Threshold:')
    print(best_threshold)
    print()
    print('Train:')
    train_flags = stats(train_pred, best_threshold)
    print_stats(train_flags, error_rate)
    print()
    print('CV:')
    cv_flags = stats(cv_pred, best_threshold)
    print_stats(cv_flags, error_rate)
    print()
    print('Test:')
    test_flags = stats(test_pred, best_threshold)
    print_stats(test_flags, error_rate)