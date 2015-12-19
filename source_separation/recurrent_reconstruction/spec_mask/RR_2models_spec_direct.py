from __future__ import division
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
import cPickle as pkl
theano.config.floatX = 'float64'


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def create_batches(n_samples, train=True):
    show = False
    if train:
        sel_mix = train_x
        sel_m = train_m
        sel_f = train_f
        batch_x = np.zeros((n_samples, max_len, n_features))
        batch_m = np.zeros((n_samples, max_len, n_features))
        batch_f = np.zeros((n_samples, max_len, n_features))
        idx = np.random.permutation(sel_mix.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x[i] = sel_mix[b:b + max_len]
            batch_m[i, :, :] = sel_m[b:b + max_len]
            batch_f[i, :, :] = sel_f[b:b + max_len]
            if show:
                plt.figure()
                plt.subplot(3, 1, 1)
                plt.imshow(batch_m[i], interpolation='nearest', aspect='auto')
                plt.subplot(3, 1, 2)
                plt.imshow(sel_m[b:b + max_len], interpolation='nearest', aspect='auto')
                plt.subplot(3, 1, 3)
                plt.imshow(sel_f[b:b + max_len], interpolation='nearest', aspect='auto')
                plt.show()
    else:
        sel_mix = test_x
        sel_m = test_m
        sel_f = test_f
        show = True
        batch_x = np.zeros((111, max_len, n_features))
        batch_m = np.zeros((111, max_len, n_features))
        batch_f = np.zeros((111, max_len, n_features))
        for i in range(111):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
            batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_x, batch_m, batch_f


data_m = loadmat('../../../data/m_f_book_SPEC.mat')['hunchback']
data_f = loadmat('../../../data/m_f_book_SPEC.mat')['screw']
data_mix = loadmat('../../../data/m_f_book_SPEC.mat')['hunchback_screw']

#########

data_mix /= np.max(np.abs(data_mix))
data_m /= np.max(np.abs(data_m))
data_f /= np.max(np.abs(data_f))
#######
n_features = data_m.shape[1]  # this time they are 512



print 'male shape {}'.format(data_m.shape)
print 'female shape {}'.format(data_f.shape)
print 'mix shape {}'.format(data_mix.shape)

max_len = 200
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

NUM_UNITS_ENC = 100
NUM_UNITS_DEC = 100

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
m_sym = T.dtensor3()
f_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()
rng = np.random.RandomState(123)


l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)

l_encoder_2_m = lasagne.layers.GRULayer(l_dec_fwd, num_units=NUM_UNITS_ENC)
l_encoder_2_f = lasagne.layers.GRULayer(l_dec_fwd, num_units=NUM_UNITS_ENC)

l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=n_features)
l_decoder_f = lasagne.layers.GRULayer(l_encoder_2_f, num_units=n_features)

output_m = lasagne.layers.get_output(l_decoder_m, inputs={l_in: x_sym})
output_f = lasagne.layers.get_output(l_decoder_f, inputs={l_in: x_sym})

# joint error function
masked_m = (output_m / (output_m + output_f + 1e-8)) * x_sym
masked_f = (output_f / (output_m + output_f + 1e-8)) * x_sym
loss_all = lasagne.objectives.squared_error(masked_m, m_sym) + \
           lasagne.objectives.squared_error(masked_f, f_sym)
# - gamma * lasagne.objectives.squared_error(masked_f, m_sym)
# - gamma * lasagne.objectives.squared_error(masked_m, f_sym)

loss_mean = T.mean(loss_all)


all_params_target_m = lasagne.layers.get_all_params([l_decoder_m])
all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean, all_params_target_m)]
all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)
updates_target_m = adam(all_grads_target_m, all_params_target_m)

all_params_target_f = lasagne.layers.get_all_params([l_decoder_f])
all_grads_target_f = [T.clip(g, -10, 10) for g in T.grad(loss_mean, all_params_target_f)]
all_grads_target_f = lasagne.updates.total_norm_constraint(all_grads_target_f, 10)
updates_target_f = adam(all_grads_target_f, all_params_target_f)

train_model_m = theano.function([x_sym, m_sym, f_sym],
                                [loss_mean, output_m, output_f],
                                updates=updates_target_m)

test_model_m = theano.function([x_sym, m_sym, f_sym],
                               [loss_mean, output_m, output_f])

num_min_batches = 100
n_batch = 100
epochs = 100

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x, batch_m, batch_f = create_batches(n_batch)
        loss_train, _, _ = train_model_m(batch_x, batch_m, batch_f)
        print 'joint loss %.10f' % loss_train

    batch_test_x, batch_test_m, batch_test_f = create_batches(100, False)
    loss_test, _, _ = test_model_m(batch_test_x, batch_test_m, batch_test_f)
    stop_time = time.time() - start_time

    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'joint loss TEST = %.10f ' % loss_test

# final test
test_x, test_m, test_f = create_batches(100, False)
l, out_m, out_f = test_model_m(test_x, test_m, test_f)
out_out_m = out_m
out_out_f = out_f

print 'JOINT TEST = %.10f' % l

savemat('RR_2models_spec_dense.mat', {'out_out_m': out_out_m,
                                      'max_m': np.max(np.abs(data_m)),
                                      'out_out_f': out_out_f,
                                      'max_f': np.max(np.abs(data_f))})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_decoder_m]),
         open('RR_2models_spec_dense_m'.format(date, NUM_UNITS_ENC), 'wb'))
pkl.dump(lasagne.layers.get_all_param_values([l_decoder_f]),
         open('RR_2models_spec_dense_f'.format(date, NUM_UNITS_ENC), 'wb'))


