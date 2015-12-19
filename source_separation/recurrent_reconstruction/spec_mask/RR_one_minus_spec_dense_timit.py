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
        #sel_mask_m = train_mask_m
        #sel_mask_f = train_mask_f
        batch_x = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_mask_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_mask_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        idx = np.random.permutation(sel_mix.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x[i] = sel_mix[b:b + max_len]
            batch_m[i, :, :] = sel_m[b:b + max_len]
            batch_f[i, :, :] = sel_f[b:b + max_len]
            #batch_mask_m[i, :, :] = sel_mask_m[b:b + max_len]
            #batch_mask_f[i, :, :] = sel_mask_f[b:b + max_len]

    else:
        sel_mix = test_x
        sel_m = test_m
        sel_f = test_f
        show = True
        Q = int(sel_mix.shape[0] / max_len)
        batch_x = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        for i in range(Q):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
            batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_x, batch_m, batch_f


train_m = loadmat('timit_classic_8.mat')['PWRa']
train_f = loadmat('timit_classic_8.mat')['PWRb']
train_x = loadmat('timit_classic_8.mat')['PWR']
test_m = loadmat('timit_classic_8.mat')['PWRatest']
test_f = loadmat('timit_classic_8.mat')['PWRbtest']
test_x = loadmat('timit_classic_8.mat')['PWRtest']

#########
# max_man = np.max(np.abs(data_m))
# max_fem = np.max(np.abs(data_f))
# data_mix /= np.max(np.abs(data_mix))
# data_m /= np.max(np.abs(data_m))
# data_f /= np.max(np.abs(data_f))
#######
n_features = train_m.shape[1]  # this time they are 512

print 'male shape {}'.format(train_m.shape)
print 'female shape {}'.format(train_f.shape)
print 'mix shape {}'.format(train_x.shape)

# # of samples is actually very high

max_len = 50
#test_mask_m = data_m[n_samples_train:]
#test_mask_f = data_f[n_samples_train:]

# savemat('check.mat',{'x':test_x,'f':test_f,'m':test_m})

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
mask_m_sym = T.tensor3()
mask_f_sym = T.tensor3()
n_sym = T.tensor3()
mask_n_sym = T.tensor3()


l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)
l_dec_bwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=True)

l_concat = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)

l_encoder_2_m = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC)
l_encoder_2_f = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC)

l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_DEC)
l_decoder_f = lasagne.layers.GRULayer(l_encoder_2_f, num_units=NUM_UNITS_DEC)

l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=tanh)
l_out_m = lasagne.layers.ReshapeLayer(l_dense_m, (-1, max_len, n_features))


output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})

# binary
#mask_m = (output_m > output_f).astype(theano.config.floatX)
#mask_f = (output_f > output_m).astype(theano.config.floatX)

print output_m.eval({x_sym: ftrain_x}).shape
# joint error function
# attempt to correct for nans
# sub = T.switch(T.isnan(comp),0.,comp)
loss_all_m = lasagne.objectives.squared_error(output_m * x_sym, m_sym) + \
             lasagne.objectives.squared_error((1. - output_m) * x_sym, f_sym)

# - gamma * lasagne.objectives.squared_error(masked_f, m_sym)
# - gamma * lasagne.objectives.squared_error(masked_m, f_sym)

loss_mean_m = T.mean(loss_all_m)


all_params_target_m = lasagne.layers.get_all_params([l_out_m])
all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean_m, all_params_target_m)]
all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)
updates_target_m = adam(all_grads_target_m, all_params_target_m)



train_model_m = theano.function([x_sym, m_sym, f_sym],
                              [loss_mean_m, output_m],
                              updates=updates_target_m)

test_model_m = theano.function([x_sym, m_sym, f_sym],
                               [loss_mean_m, output_m])


num_min_batches = 100
n_batch = 100
epochs = 75

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x, batch_m, batch_f = create_batches(n_batch)
        loss_train_m, _ = train_model_m(batch_x, batch_m, batch_f)
        print 'M loss %.10f' % loss_train_m,
    batch_test_x, batch_test_m, batch_test_f = create_batches(100, False)
    loss_test_m, out_m = test_model_m(batch_test_x, batch_test_m, batch_test_f)
    stop_time = time.time() - start_time

    # filt_m = out_m * batch_test_x
    # filt_f = out_f * batch_test_x
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(batch_test_m[0], interpolation='nearest', aspect='auto')
    # plt.subplot(2, 2, 2)
    # plt.imshow(filt_m[0], interpolation='nearest', aspect='auto')
    # plt.subplot(2, 2, 3)
    # plt.imshow(batch_test_f[0], interpolation='nearest', aspect='auto')
    # plt.subplot(2, 2, 4)
    # plt.imshow(filt_f[0], interpolation='nearest', aspect='auto')
    # plt.show()
    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'M loss TEST = %.10f ' % loss_test_m,

# final test
test_x, test_m, test_f = create_batches(100, False)
l_m, out_m = test_model_m(test_x, test_m, test_f)
out_out_m = out_m

print 'M TEST = %.10f' % l_m,

savemat('RR_2models_spec_dense_bwd.mat', {'out_out_m': out_out_m,
                                       })
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_out_m]),
         open('RR_2models_spec_dense_m_bwd'.format(date, NUM_UNITS_ENC), 'wb'))


