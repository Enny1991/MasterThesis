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
        batch_t = np.zeros((n_samples, max_len, n_features * 2))
        idx = np.random.permutation(sel_mix.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x[i] = sel_mix[b:b + max_len]
            batch_t[i, :, :n_features] = sel_m[b:b + max_len]
            batch_t[i, :, n_features:] = sel_f[b:b + max_len]
            if show:
                plt.figure()
                plt.subplot(3, 1, 1)
                plt.imshow(batch_t[i], interpolation='nearest', aspect='auto')
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
        batch_t = np.zeros((111, max_len, n_features * 2))
        for i in range(111):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_t[i, :, :n_features] = sel_m[i*max_len:((i+1)*max_len)]
            batch_t[i, :, n_features:] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_x, batch_t


data_m = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback']
data_f = loadmat('../../../data/m_f_book_POLAR.mat')['screw']
data_mix = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback_screw']

data_mix[:, :16] /= np.max(np.abs(data_mix[:, :16]))
data_mix[:, 16:] /= np.pi
data_m[:, :16] /= np.max(np.abs(data_m[:, :16]))
data_m[:, 16:] /= np.pi
data_f[:, :16] /= np.max(np.abs(data_f[:, :16]))
data_f[:, 16:] /= np.pi
print np.min(data_mix)
print np.max(data_mix)
# find max

n_features = data_m.shape[1]

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

savemat('check.mat',{'x':test_x,'f':test_f,'m':test_m})

# create teaching signal

n_batch = 40
n_test = 60

ftrain_x, ftrain_t = create_batches(10)
ftest_x, ftest_t = create_batches(10, False)


print ftrain_x.shape
print ftrain_t.shape
print ftest_x.shape
print ftest_t.shape

NUM_UNITS_ENC = 100
NUM_UNITS_DEC = 100

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()
rng = np.random.RandomState(123)

l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

print lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)
l_dec_fwd_1 = lasagne.layers.GRULayer(l_dec_fwd, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)

l_decoder = lasagne.layers.GRULayer(l_dec_fwd_1, num_units=n_features*2)

output = lasagne.layers.get_output(l_decoder, inputs={l_in: x_sym})

loss_all_target = lasagne.objectives.squared_error(output, t_sym)

loss_mean_target = T.mean(loss_all_target)

all_params_target = lasagne.layers.get_all_params([l_decoder])
all_grads_target = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target, all_params_target)]
all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 10)
updates_target = adam(all_grads_target, all_params_target)

train_model = theano.function([x_sym, t_sym],
                              [loss_mean_target, output],
                              updates=updates_target)

test_model = theano.function([x_sym, t_sym],
                             [loss_mean_target, output])


num_min_batches = 100
n_batch = 100
epochs = 100

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x, batch_t = create_batches(n_batch)
        loss_train_target, out = train_model(batch_x, batch_t)
        print 'loss batch %.10f' % loss_train_target

    batch_test_x, batch_test_t = create_batches(100, False)
    loss_test, out = test_model(batch_test_x, batch_test_t)
    stop_time = time.time() - start_time

    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'loss_train_target = %.10f \nloss_test_target = %.10f' % (loss_train_target, loss_test)

# final test
test_x, test_t = create_batches(100, False)
l, out = test_model(test_x, test_t)
out_out = out

print 'TEST = %.6f' % l

savemat('RR_polar_direct.mat', {'out_out': out_out,
                                'max_m': np.max(np.abs(data_m[:, :16])),
                                'max_f': np.max(np.abs(data_f[:, :16]))})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_decoder]),
          open('RR_polar_direct'.format(date, NUM_UNITS_ENC), 'wb'))


