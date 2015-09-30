import theano
import theano.tensor as T
from theano import dot
from theano.tensor.nnet import sigmoid as sigm
from theano.tensor import tanh
from theano.tensor.nnet import softmax
from theano.tensor.nnet import categorical_crossentropy
import numpy as np
import sys
import os
import helper

home_dir = os.path.expanduser('~')

### BEGING SETTINGS ###

# text corpus 
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')

# work dir will contain pickled lstm weights and pickled list of training errors
work_dir = os.path.join(home_dir, 'training_lstm_lm')

# must be set to load corresponding weights from work_dir
weights_file_name = ''

# number of neurons in hidden layer of lstm
hidden_layer_size = 1024

# 40% of occurrences of these tokens will be excluded from training corpus for cv and test
preserve_tokens = ['than', 'then', 'except', 'accept', 'well', 'good']

# order matters: algorithm will generate rules for occurrences of first word in list
confusion_set = ['than', 'then']

### END SETTINGS ###

# check if weights info has been set

if not weights_file_name:
    print('ERROR: weights_file_name must be set!')
    sys.exit(1)
    
# init    
    
with open(corpus_file) as f:
    sents = [[twp.split('|')[0].lower() for twp in line.split()] for line in f]
    
embeddings = helper.char_embeddings(sents)

print('init network and compile theano functions...')

h_size = hidden_layer_size # hidden size

W_xi, W_hi, W_ci, b_i, \
W_xf, W_hf, W_cf, b_f, \
W_xc, W_hc, b_c, \
W_xo, W_ho, W_co, b_o, \
W_hy, b_y = helper.load_states(weights_file_name, work_dir)

S_h = helper.init_zero_vec(h_size) # init values for hidden units
S_c = helper.init_zero_vec(h_size) # init values for cell units

S_x = T.matrix() # inputs

# BEGIN code inspired by Christian Herta 
# http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php

def step(S_x, S_h, S_c, 
         W_xi, W_hi, W_ci, b_i, 
         W_xf, W_hf, W_cf, b_f, 
         W_xc, W_hc, b_c, 
         W_xo, W_ho, W_co, b_o, 
         W_hy, b_y):
    
    S_i = sigm(dot(S_x, W_xi) + dot(S_h, W_hi) + dot(S_c, W_ci) + b_i)
    S_f = sigm(dot(S_x, W_xf) + dot(S_h, W_hf) + dot(S_c, W_cf) + b_f)
    S_c = S_f * S_c + S_i * tanh(dot(S_x, W_xc) + dot(S_h, W_hc) + b_c) 
    S_o = sigm(dot(S_x, W_xo) + dot(S_h, W_ho) + dot(S_c, W_co) + b_o)
    S_h = S_o * tanh(S_c)
    S_y = dot(S_h, W_hy) + b_y
    
    return [S_h, S_c, S_y]

# scan loops through input sequence and applies step function to each time step

(S_h_r, S_c_r, S_y_r ), _ = theano.scan(fn = step,
                                        sequences = S_x,
                                        outputs_info = [S_h, S_c, None],
                                        non_sequences = [W_xi, W_hi, W_ci, b_i, 
                                                         W_xf, W_hf, W_cf, b_f, 
                                                         W_xc, W_hc, b_c, 
                                                         W_xo, W_ho, W_co, b_o, 
                                                         W_hy, b_y])
                                                         
predict = theano.function(inputs=[S_x], 
                          outputs=S_y_r,
                          allow_input_downcast=True)

# sampling theano functions

S_h_v = T.vector()
S_c_v = T.vector()
    
S_h_s, S_c_s, S_y_s = step(S_x, S_h_v, S_c_v, 
                           W_xi, W_hi, W_ci, b_i, 
                           W_xf, W_hf, W_cf, b_f, 
                           W_xc, W_hc, b_c, 
                           W_xo, W_ho, W_co, b_o, 
                           W_hy, b_y)

sampling = theano.function(inputs = [S_x, S_h_v, S_c_v], 
                           outputs = [S_h_s, S_c_s, S_y_s],
                           allow_input_downcast=True)
                                                         
# END code inspired by Christian Herta

# define functions
                               
def apply_sampling(embeddings, hid, start='S', end='E', t=1.0):
    x = embeddings[start]
    e = embeddings[end]
    
    S_x = helper.index_to_vec(start, embeddings)
    
    S_h = np.zeros(hid, dtype=theano.config.floatX)
    S_c = np.zeros(hid, dtype=theano.config.floatX)
    
    sampled = [x]
    
    counter = 0
    
    while x != e:
        S_x = np.reshape(S_x, (1, -1))
        S_h, S_c, S_y = sampling(S_x, S_h.flatten(), S_c.flatten())
        S_y = helper.t_softmax(S_y.flatten(), t=t)
        S_x = np.random.multinomial(n=1, pvals=S_y)
        x = helper.vec_to_index(S_x)
        sampled.append(x)
        if counter == 1000:
            break
        counter += 1
        
    return sampled[1:-1]

def resample(embeddings, h_size, min_tokens=0, trials=100, t=1.0):
    for i in range(trials):
        try:
            sample = apply_sampling(embeddings, h_size, t=t)
            sample = helper.sampled_chars_to_string(sample, embeddings)
            if len(sample.split()) < min_tokens:
                continue
            return sample
        except:
            pass
    return 'NO SAMPLE IN %d STEPS' % trials    

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
    log_p = 0
    sequence = list(' '.join(sentence)) + ['E']
    for c, p in zip(sequence, predictions):
        probabilities = helper.t_softmax(p, t=t)
        index = embeddings[c]
        log_p += np.log(probabilities[index])
    return log_p
    
def lm_predictions(samples_0, samples_1, confusion_set, embeddings):
    predictions_0_correct = []
    predictions_0_wrong = []
    for sentence, index in samples_0:
        S_x, Y = helper.char_sequence(sentence, embeddings)
        S_y = predict(S_x)
        predictions_0_correct.append(S_y)
        sentence[index] = confusion_set[1]
        S_x, Y = helper.char_sequence(sentence, embeddings)
        S_y = predict(S_x)
        predictions_0_wrong.append(S_y)
    
    predictions_1_correct = []
    predictions_1_wrong = []
    for sentence, index in samples_1:
        S_x, Y = helper.char_sequence(sentence, embeddings)
        S_y = predict(S_x)
        predictions_1_correct.append(S_y)
        sentence[index] = confusion_set[0]
        S_x, Y = helper.char_sequence(sentence, embeddings)
        S_y = predict(S_x)
        predictions_1_wrong.append(S_y)
        
    return [predictions_0_correct, predictions_0_wrong, predictions_1_correct, predictions_1_wrong]

# generate samples

min_tokens = 5

print('genrate samples')
print('minimum number of tokens per sample: ', min_tokens)
print()
for t in [0.8, 1.0, 1.2]:
    print('temperature: ', t)
    print()
    for i in range(20):
        print(resample(embeddings, h_size, min_tokens=min_tokens, trials=100, t=t))
    print()
    
# apply lstm language model to confusion_set

print('apply lstm language model to confusion set...')
print()

cv_0 = list(helper.acs(sents, preserve_tokens, cv_token=confusion_set[0]))
cv_1 = list(helper.acs(sents, preserve_tokens, cv_token=confusion_set[1]))
test_0 = list(helper.acs(sents, preserve_tokens, test_token=confusion_set[0]))
test_1 = list(helper.acs(sents, preserve_tokens, test_token=confusion_set[1]))

cv_predictions = lm_predictions(cv_0, cv_1, confusion_set, embeddings)
test_predictions = lm_predictions(test_0, test_1, confusion_set, embeddings)

# find best temperature and threshold combination and print results

lm_results(cv_predictions, test_predictions, cv_0, cv_1, test_0, test_1, embeddings, confusion_set)

print()
print('done')