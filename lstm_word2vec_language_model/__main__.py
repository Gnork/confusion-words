__author__ = 'Christoph Jansen, HTW Berlin'

import theano
import theano.tensor as T
from theano import dot
from theano.tensor.nnet import sigmoid as sigm
from theano.tensor import tanh
from theano.tensor.nnet import softmax
from theano.tensor.nnet import categorical_crossentropy
import os
import numpy as np
from datetime import datetime
import helper

home_dir = os.path.expanduser('~')

### BEGING SETTINGS ###

# text corpus 
corpus_file = os.path.join(home_dir, 'brown_tagged.txt')

# word2vec embeddings
embeddings_file = os.path.join(home_dir, 'normalized_embeddings.bin')

# work dir will contain pickled lstm weights and pickled list of training errors
work_dir = os.path.join(home_dir, 'training_lstm_w2v_lm')

# if training should be continued from existing weights, timestamp, start_epoch and start_iteration must be given
# every training is identified by a generated timestamp
# else set values to None
timestamp = None        # string
start_epoch = None      # int
start_iteration = None  # int

# number of neurons in hidden layer of lstm
hidden_layer_size = 512

# 40% of occurrences of these tokens will be excluded from training corpus for cv and test
preserve_tokens = ['than', 'then', 'except', 'accept', 'well', 'good']

# number of training epochs 
# complete corpus will be given to lstm for training once per epoch 
max_epochs = 1

# after training lstm language model will be applied to this confusion set
# order matters: algorithm will generate rules for occurrences of first word in list
confusion_set = ['than', 'then']

# minimum occurence of tokens in training data 
# tokens with less occurences will be substituted to 'U' for unknown
# 'U' can also serve as substitute for unseen tokens at test time
min_occurrence = 20

### END SETTINGS ###
       
# init

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
    
with open(corpus_file) as f:
    sents = [[helper.normalization(twp.split('|')[0].lower()) for twp in line.split()] for line in f]

train_sents = list(helper.acs(sents, preserve_tokens)) 
token_embeddings = helper.TokenEmbeddings(train_sents, min_occurrence)
w2v_embeddings = helper.Word2VecEmbeddings(embeddings_file)

if timestamp and start_epoch and start_iteration:
    errors = helper.load_errors('%s-%d-%d.errors' % (timestamp, start_epoch, start_iteration), work_dir)
    load_weights = '%s-%d-%d.weights' % (timestamp, start_epoch, start_iteration)
    print('init previous states...')
    print('timestamp: ', timestamp)
    print('start_epoch: ', start_epoch)
    print('start_iteration: ', start_iteration)
else:
    errors = []
    start_epoch = 0
    start_iteration = 0
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    load_weights = None
    print('init new states...')
    print('timestamp: ', timestamp)
print()

# initialize lstm weights
    
inp = w2v_embeddings.embeddings_size # input/output size
hid = hidden_layer_size # hidden size
out = token_embeddings.num_tokens

if not load_weights:

    W_xi = helper.init_weights((inp, hid))
    W_hi = helper.init_weights((hid, hid))
    W_ci = helper.init_weights((hid, hid))
    b_i = helper.init_zero_vec(hid)

    W_xf = helper.init_weights((inp, hid))
    W_hf = helper.init_weights((hid, hid))
    W_cf = helper.init_weights((hid, hid))
    b_f = helper.init_zero_vec(hid)

    W_xc = helper.init_weights((inp, hid))  
    W_hc = helper.init_weights((hid, hid))
    b_c = helper.init_zero_vec(hid)

    W_xo = helper.init_weights((inp, hid))
    W_ho = helper.init_weights((hid, hid))
    W_co = helper.init_weights((hid, hid))
    b_o = helper.init_zero_vec(hid)

    W_hy = helper.init_weights((hid, out))
    b_y = helper.init_zero_vec(out)

else:
    W_xi, W_hi, W_ci, b_i, \
    W_xf, W_hf, W_cf, b_f, \
    W_xc, W_hc, b_c, \
    W_xo, W_ho, W_co, b_o, \
    W_hy, b_y = helper.load_states(load_weights, work_dir)
    
# LSTM code

S_h = helper.init_zero_vec(hid) # init values for hidden units
S_c = helper.init_zero_vec(hid) # init values for cell units

S_x = T.matrix() # inputs
Y = T.matrix() # targets

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
                                                         
# END code inspired by Christian Herta

# cost and gradient descent

cost = T.mean(categorical_crossentropy(softmax(S_y_r), Y))

def gradient_descent(cost, weights, lr=0.05):
    grads = T.grad(cost=cost, wrt=weights)
    updates = []
    for w, g in zip(weights, grads):
        updates.append([w, w - lr * g])
    return updates

updates = gradient_descent(cost, 
                           [W_xi, W_hi, W_ci, b_i, 
                            W_xf, W_hf, W_cf, b_f, 
                            W_xc, W_hc, b_c, 
                            W_xo, W_ho, W_co, b_o, 
                            W_hy, b_y])

# training theano function

train = theano.function(inputs=[S_x, Y], 
                        outputs=cost, 
                        updates=updates, 
                        allow_input_downcast=True)

# prediction theano function

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

# sampling python functions
                               
def apply_sampling(token_embeddings, w2v_embeddings, hid, start='S', end='E', t=1.0, max_tokens=50):
    tokens = token_embeddings.tokens
    
    S_x = w2v_embeddings.token_to_vec(start)
    
    S_h = np.zeros(hid, dtype=theano.config.floatX)
    S_c = np.zeros(hid, dtype=theano.config.floatX)
    
    sampled_tokens = [start]
    
    counter = 0
    
    while sampled_tokens[-1] != end:
        if counter == max_tokens:
            sampled_tokens.append(end)
            break
        
        S_x = helper.theano_cast(S_x)    
        S_x = np.reshape(S_x, (1, -1))
        S_h, S_c, S_y = sampling(S_x, S_h.flatten(), S_c.flatten())
        S_y = S_y.flatten()
        distribution = helper.t_softmax(S_y, t=t)
        S_x = np.random.multinomial(n=1, pvals=distribution)
        idx = helper.vec_to_index(S_x)
        sampled_token = tokens[idx]
        sampled_tokens.append(sampled_token)
        S_x = w2v_embeddings.token_to_vec(sampled_token)
        
        counter += 1
        
    return sampled_tokens[1:-1]

def resample(token_embeddings, w2v_embeddings, hid, min_tokens=0, max_tokens=50, trials=100, t=1.0):
    for i in range(trials):
        try:
            sample = apply_sampling(token_embeddings, w2v_embeddings, hid, t=t, max_tokens=max_tokens)
            if len(sample) < min_tokens:
                continue
            return ' '.join(sample)
        except:
            pass
    return 'NO SAMPLE IN %d STEPS' % trials

# training

print('start training...')
print()
log_steps = 500
save_steps = 5000

weights_changed = False

for e in range(max_epochs):
    if e < start_epoch:
        continue
    error = 0
    for i, (inp, tar) in enumerate(helper.token_sequence_generator(train_sents, token_embeddings, w2v_embeddings)):
        
        if e == start_epoch and i < start_iteration:
            continue
        
        cost = train(inp, tar)
        error += cost
        weights_changed = True
        
        if (i+1) % log_steps == 0:
            error /= log_steps
            errors.append(error)
            print('epoch: %d\titerations: %d\terror: %f' %(e, (i+1), error))
            print(resample(token_embeddings, w2v_embeddings, hid))
            print()
            error = 0
            
        if (i+1) % save_steps == 0:
            helper.save_states([W_xi, W_hi, W_ci, b_i, 
                         W_xf, W_hf, W_cf, b_f, 
                         W_xc, W_hc, b_c, 
                         W_xo, W_ho, W_co, b_o, 
                         W_hy, b_y], 
                        '%s-%d-%d.weights' % (timestamp, e, (i+1)), work_dir)
            helper.save_errors(errors, '%s-%d-%d.errors' % (timestamp, e, (i+1)), work_dir)
            weights_changed = False
            print('weights saved:')
            print('%s-%d-%d.weights' % (timestamp, e, (i+1)))
            print('errors saved:')
            print('%s-%d-%d.errors' % (timestamp, e, (i+1)))
            print()

print('end training')
print()

# save current weights if training has been performed

if weights_changed:
    helper.save_states([W_xi, W_hi, W_ci, b_i, 
                        W_xf, W_hf, W_cf, b_f, 
                        W_xc, W_hc, b_c, 
                        W_xo, W_ho, W_co, b_o, 
                        W_hy, b_y], '%s-%d-%d.weights' % (timestamp, e, (i+1)), work_dir)
    
    helper.save_errors(errors, '%s-%d-%d.errors' % (timestamp, e, (i+1)), work_dir)
    
    print('final weights saved:')
    print('%s-%d-%d.weights' % (timestamp, e, (i+1)))
    print('final errors saved:')
    print('%s-%d-%d.errors' % (timestamp, e, (i+1)))
    print()

# generate samples

min_tokens = 5
max_tokens = 50

num_samples = 20

print('genrate samples')
print('minimum number of tokens per sample: ', min_tokens)
print()
for t in [0.8, 1.0, 1.2]:
    print('temperature: ', t)
    print()
    for i in range(num_samples):
        print(resample(token_embeddings, w2v_embeddings, hid, min_tokens=min_tokens, max_tokens=max_tokens, trials=100, t=t))
    print()