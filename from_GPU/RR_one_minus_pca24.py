from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
#import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
import cPickle as pkl


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def create_batches(n_samples, train=True):
    show = False
    if train:
        sel_mix = train_x
        sel_m = train_m
        sel_f = train_f
        batch_x = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        idx = np.random.permutation(sel_mix.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x[i] = sel_mix[b:b + max_len]
            batch_m[i, :, :] = sel_m[b:b + max_len]
            batch_f[i, :, :] = sel_f[b:b + max_len]

    else:
        sel_mix = test_x
        sel_m = test_m
        sel_f = test_f
        show = True
        batch_x = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        for i in range(111):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
            batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_x, \
           batch_m, \
           batch_f


data_m = loadmat('m_f_book_24.mat')['hunchback']
data_f = loadmat('m_f_book_24.mat')['screw']
data_mix = loadmat('m_f_book_24.mat')['hunchback_screw']
# data_m = loadmat('../../../data/m_f_book_24.mat')['hunchback']
# data_f = loadmat('../../../data/m_f_book_24.mat')['screw']
# data_mix = loadmat('../../../data/m_f_book_24.mat')['hunchback_screw']

data_mix /= np.max(np.abs(data_mix))
data_f /= np.max(np.abs(data_f))
data_m /= np.max(np.abs(data_m))

n_features = data_m.shape[1]



print 'male shape {}'.format(data_m.shape)
print 'female shape {}'.format(data_f.shape)
print 'mix shape {}'.format(data_mix.shape)

max_len = 100
n_samples_train = 360000
n_samples_test = 22200
# # of samples is actually very high

# divide
train_x = data_mix[:n_samples_train]
train_m = data_m[:n_samples_train]
train_f = data_f[:n_samples_train]
test_x = data_mix[n_samples_train:]
test_m = data_m[n_samples_train:]
test_f = data_f[n_samples_train:]

savemat('check.mat',{'x':test_x,'f':test_f,'m':test_m})

# create teaching signal

n_batch = 40
n_test = 60

ftrain_x, ftrain_m, ftrain_f = create_batches(10)
ftest_x, ftest_m, ftest_f = create_batches(10, False)


print ftrain_x.shape
print ftrain_m.shape
print ftest_x.shape
print ftest_m.shape

NUM_UNITS_ENC = 200
NUM_UNITS_DEC = 200

x_sym = T.tensor3()
mask_x_sym = T.matrix()
m_sym = T.tensor3()
f_sym = T.tensor3()
mask_t_sym = T.tensor3()
n_sym = T.tensor3()
mask_n_sym = T.tensor3()


l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)
l_dec_bwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=True)

l_concat = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)

l_encoder_2_m = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC)


l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_DEC)

l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=rectify)
l_out_m = lasagne.layers.ReshapeLayer(l_dense_m, (-1, max_len, n_features))

output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})

loss_all_target_m = T.mean(lasagne.objectives.squared_error(output_m * x_sym, m_sym)) + \
                    T.mean(lasagne.objectives.squared_error((1. - output_m) * x_sym, f_sym))

loss_mean_target_m = loss_all_target_m

all_params_target_m = lasagne.layers.get_all_params([l_out_m])
all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target_m, all_params_target_m)]
all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)
updates_target_m = adam(all_grads_target_m, all_params_target_m)

train_model_m = theano.function([x_sym, m_sym, f_sym],
                              [loss_mean_target_m, output_m],
                              updates=updates_target_m)

test_model_m = theano.function([x_sym, m_sym, f_sym],
                             [loss_mean_target_m, output_m])





num_min_batches = 100
n_batch = 150
epochs = 100

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x, batch_m, batch_f = create_batches(n_batch)
        loss_train_m, _ = train_model_m(batch_x, batch_m, batch_f)
        print 'loss male %.10f' % loss_train_m

    batch_test_x, batch_test_m, batch_test_f = create_batches(100, False)
    loss_test_m, out_m = test_model_m(batch_test_x, batch_test_m, batch_test_f)
    stop_time = time.time() - start_time

    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'loss male TEST = %.10f' % (loss_test_m)

# final test
test_x, test_m, test_f = create_batches(100, False)
l_m, out_m = test_model_m(test_x, test_m, test_f)
out_out_m = out_m

print 'TEST M = %.6f' % l_m

savemat('RR_one_minus_PCA24_dense.mat', {'out_out_m': out_out_m,
                                       'max_m': np.max(np.abs(data_m))})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_out_m]),
         open('RR_one_minus_PCA24'.format(date, NUM_UNITS_ENC), 'wb'))



