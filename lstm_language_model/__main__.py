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

# work dir will contain pickled lstm weights and pickled list of training errors
work_dir = os.path.join(home_dir, 'training_lstm_lm')

# if training should be continued from existing weights, timestamp, start_epoch and start_iteration must be given
# every training is identified by a generated timestamp
# else set values to None
timestamp = None        # string
start_epoch = None      # int
start_iteration = None  # int

# number of neurons in hidden layer of lstm
hidden_layer_size = 1024

# 40% of occurrences of these tokens will be excluded from training corpus for cv and test
preserve_tokens = ['than', 'then', 'except', 'accept', 'well', 'good']

# number of training epochs 
# complete corpus will be given to lstm for training once per epoch 
max_epochs = 1

# order matters: algorithm will generate rules for occurrences of first word in list
confusion_set = ['than', 'then']

### END SETTINGS ###
       
# init

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
    
with open(corpus_file) as f:
    sents = [[twp.split('|')[0].lower() for twp in line.split()] for line in f]
    
embeddings = helper.char_embeddings(sents)

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
    
io_size = len(embeddings) # input/output size
h_size = hidden_layer_size # hidden size

if not load_weights:

    W_xi = helper.init_weights((io_size, h_size))
    W_hi = helper.init_weights((h_size, h_size))
    W_ci = helper.init_weights((h_size, h_size))
    b_i = helper.init_zero_vec(h_size)

    W_xf = helper.init_weights((io_size, h_size))
    W_hf = helper.init_weights((h_size, h_size))
    W_cf = helper.init_weights((h_size, h_size))
    b_f = helper.init_zero_vec(h_size)

    W_xc = helper.init_weights((io_size, h_size))  
    W_hc = helper.init_weights((h_size, h_size))
    b_c = helper.init_zero_vec(h_size)

    W_xo = helper.init_weights((io_size, h_size))
    W_ho = helper.init_weights((h_size, h_size))
    W_co = helper.init_weights((h_size, h_size))
    b_o = helper.init_zero_vec(h_size)

    W_hy = helper.init_weights((h_size, io_size))
    b_y = helper.init_zero_vec(io_size)

else:
    W_xi, W_hi, W_ci, b_i, \
    W_xf, W_hf, W_cf, b_f, \
    W_xc, W_hc, b_c, \
    W_xo, W_ho, W_co, b_o, \
    W_hy, b_y = helper.load_states(load_weights, work_dir)
    
# LSTM code

S_h = helper.init_zero_vec(h_size) # init values for hidden units
S_c = helper.init_zero_vec(h_size) # init values for cell units

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

# training

print('start training...')
print()
log_steps = 100
save_steps = 100

weights_changed = False

for e in range(max_epochs):
    if e < start_epoch:
        continue
    error = 0
    for i, (inp, tar) in enumerate(helper.char_sequence_generator(helper.acs(sents, preserve_tokens), embeddings)):
        
        if e == start_epoch and i < start_iteration:
            continue
        
        cost = train(inp, tar)
        error += cost
        weights_changed = True
        
        if (i+1) % log_steps == 0:
            error /= log_steps
            errors.append(error)
            print('epoch: %d\titerations: %d\terror: %f' %(e, (i+1), error))
            print(resample(embeddings, h_size))
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
    
print('done')