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
        batch_x_a = np.zeros((n_samples, max_len, n_features))
        batch_m_a = np.zeros((n_samples, max_len, n_features))
        batch_f_a = np.zeros((n_samples, max_len, n_features))
        batch_x_p = np.zeros((n_samples, max_len, n_features))
        batch_m_p = np.zeros((n_samples, max_len, n_features))
        batch_f_p = np.zeros((n_samples, max_len, n_features))
        idx = np.random.permutation(train_x_a.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x_a[i] = train_x_a[b:b + max_len]
            batch_m_a[i, :, :] = train_m_a[b:b + max_len]
            batch_f_a[i, :, :] = train_f_a[b:b + max_len]
            batch_x_p[i] = train_x_p[b:b + max_len]
            batch_m_p[i, :, :] = train_m_p[b:b + max_len]
            batch_f_p[i, :, :] = train_f_p[b:b + max_len]
    else:
        batch_x_a = np.zeros((111, max_len, n_features))
        batch_m_a = np.zeros((111, max_len, n_features))
        batch_f_a = np.zeros((111, max_len, n_features))
        batch_x_p = np.zeros((111, max_len, n_features))
        batch_m_p = np.zeros((111, max_len, n_features))
        batch_f_p = np.zeros((111, max_len, n_features))
        for i in range(111):
            batch_x_a[i] = test_x_a[i*max_len:((i+1)*max_len)]
            batch_m_a[i, :, :] = test_m_a[i*max_len:((i+1)*max_len)]
            batch_f_a[i, :, :] = test_f_a[i*max_len:((i+1)*max_len)]
            batch_x_p[i] = test_x_p[i*max_len:((i+1)*max_len)]
            batch_m_p[i, :, :] = test_m_p[i*max_len:((i+1)*max_len)]
            batch_f_p[i, :, :] = test_f_p[i*max_len:((i+1)*max_len)]
    return batch_x_a, batch_m_a, batch_f_a, batch_x_p, batch_m_p, batch_f_p


data_m_a = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback_a']
data_f_a = loadmat('../../../data/m_f_book_POLAR.mat')['screw_a']
data_mix_a = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback_screw_a']
data_m_p = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback_p']
data_f_p = loadmat('../../../data/m_f_book_POLAR.mat')['screw_p']
data_mix_p = loadmat('../../../data/m_f_book_POLAR.mat')['hunchback_screw_p']

#########
data_m_a /= np.max(np.abs(data_m_a))
data_f_a /= np.max(np.abs(data_f_a))
data_mix_a /= np.max(np.abs(data_mix_a))
data_m_p /= np.max(np.abs(data_m_p))
data_f_p /= np.max(np.abs(data_f_p))
data_mix_p /= np.max(np.abs(data_mix_p))
#######
n_features = data_m_a.shape[1]



max_len = 200
n_samples_train = 360000
n_samples_test = 22200
# # of samples is actually very high

# divide
train_x_a = data_mix_a[:n_samples_train]
train_m_a = data_m_a[:n_samples_train]
train_f_a = data_f_a[:n_samples_train]
test_x_a = data_mix_a[n_samples_train:]
test_m_a = data_m_a[n_samples_train:]
test_f_a = data_f_a[n_samples_train:]

train_x_p = data_mix_p[:n_samples_train]
train_m_p = data_m_p[:n_samples_train]
train_f_p = data_f_p[:n_samples_train]
test_x_p = data_mix_p[n_samples_train:]
test_m_p = data_m_p[n_samples_train:]
test_f_p = data_f_p[n_samples_train:]


# create teaching signal

n_batch = 40
n_test = 60

ftrain_x_a, ftrain_m_a, ftrain_f_a, ftrain_x_p, ftrain_m_p, ftrain_f_p = create_batches(10)
ftest_x_a, ftest_m_a, ftest_f_a, ftest_x_p, ftest_m_p, ftest_f_p = create_batches(10, False)


print ftrain_x_a.shape
print ftrain_m_p.shape
print ftest_x_p.shape
print ftest_m_a.shape

NUM_UNITS_ENC = 100
NUM_UNITS_DEC = 100

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()
rng = np.random.RandomState(123)


# ABS #####################################################################

l_in_a = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd_a = lasagne.layers.GRULayer(l_in_a, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)

l_encoder_2_m_a = lasagne.layers.GRULayer(l_dec_fwd_a, num_units=NUM_UNITS_ENC)
l_encoder_2_f_a = lasagne.layers.GRULayer(l_dec_fwd_a, num_units=NUM_UNITS_ENC)

l_decoder_m_a = lasagne.layers.GRULayer(l_encoder_2_m_a, num_units=NUM_UNITS_DEC)
l_decoder_f_a = lasagne.layers.GRULayer(l_encoder_2_f_a, num_units=NUM_UNITS_DEC)

l_reshape_m_a = lasagne.layers.ReshapeLayer(l_decoder_m_a, (-1, NUM_UNITS_DEC))
l_dense_m_a = lasagne.layers.DenseLayer(l_reshape_m_a, num_units=n_features, nonlinearity=identity)
l_out_m_a = lasagne.layers.ReshapeLayer(l_dense_m_a, (-1, max_len, n_features))

l_reshape_f_a = lasagne.layers.ReshapeLayer(l_decoder_f_a, (-1, NUM_UNITS_DEC))
l_dense_f_a = lasagne.layers.DenseLayer(l_reshape_f_a, num_units=n_features, nonlinearity=identity)
l_out_f_a = lasagne.layers.ReshapeLayer(l_dense_f_a, (-1, max_len, n_features))


output_m_a = lasagne.layers.get_output(l_out_m_a, inputs={l_in_a: x_sym})
output_f_a = lasagne.layers.get_output(l_out_f_a, inputs={l_in_a: x_sym})

loss_all_target_m_a = lasagne.objectives.squared_error(output_m_a, t_sym)

loss_mean_target_m_a = T.mean(loss_all_target_m_a)

loss_all_target_f_a = lasagne.objectives.squared_error(output_f_a, t_sym)

loss_mean_target_f_a = T.mean(loss_all_target_f_a)

all_params_target_m_a = lasagne.layers.get_all_params([l_out_m_a])
all_grads_target_m_a = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target_m_a, all_params_target_m_a)]
all_grads_target_m_a = lasagne.updates.total_norm_constraint(all_grads_target_m_a, 10)
updates_target_m_a = adam(all_grads_target_m_a, all_params_target_m_a)

all_params_target_f_a = lasagne.layers.get_all_params([l_out_f_a])
all_grads_target_f_a = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target_f_a, all_params_target_f_a)]
all_grads_target_f_a = lasagne.updates.total_norm_constraint(all_grads_target_f_a, 10)
updates_target_f_a = adam(all_grads_target_f_a, all_params_target_f_a)

train_model_m_a = theano.function([x_sym, t_sym],
                              [loss_mean_target_m_a, output_m_a],
                              updates=updates_target_m_a)

test_model_m_a = theano.function([x_sym, t_sym],
                             [loss_mean_target_m_a, output_m_a])


train_model_f_a = theano.function([x_sym, t_sym],
                              [loss_mean_target_f_a, output_f_a],
                              updates=updates_target_f_a)

test_model_f_a = theano.function([x_sym, t_sym],
                               [loss_mean_target_f_a, output_f_a])

# PHASE##############################################################################

l_in_p = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd_p = lasagne.layers.GRULayer(l_in_p, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)

l_encoder_2_m_p = lasagne.layers.GRULayer(l_dec_fwd_p, num_units=NUM_UNITS_ENC)
l_encoder_2_f_p = lasagne.layers.GRULayer(l_dec_fwd_p, num_units=NUM_UNITS_ENC)

l_decoder_m_p = lasagne.layers.GRULayer(l_encoder_2_m_p, num_units=NUM_UNITS_DEC)
l_decoder_f_p = lasagne.layers.GRULayer(l_encoder_2_f_p, num_units=NUM_UNITS_DEC)

l_reshape_m_p = lasagne.layers.ReshapeLayer(l_decoder_m_p, (-1, NUM_UNITS_DEC))
l_dense_m_p = lasagne.layers.DenseLayer(l_reshape_m_p, num_units=n_features, nonlinearity=identity)
l_out_m_p = lasagne.layers.ReshapeLayer(l_dense_m_p, (-1, max_len, n_features))

l_reshape_f_p = lasagne.layers.ReshapeLayer(l_decoder_f_p, (-1, NUM_UNITS_DEC))
l_dense_f_p = lasagne.layers.DenseLayer(l_reshape_f_p, num_units=n_features, nonlinearity=identity)
l_out_f_p = lasagne.layers.ReshapeLayer(l_dense_f_p, (-1, max_len, n_features))


output_m_p = lasagne.layers.get_output(l_out_m_p, inputs={l_in_p: x_sym})
output_f_p = lasagne.layers.get_output(l_out_f_p, inputs={l_in_p: x_sym})

loss_all_target_m_p = lasagne.objectives.squared_error(output_m_p, t_sym)

loss_mean_target_m_p = T.mean(loss_all_target_m_p)

loss_all_target_f_p = lasagne.objectives.squared_error(output_f_p, t_sym)

loss_mean_target_f_p = T.mean(loss_all_target_f_p)

all_params_target_m_p = lasagne.layers.get_all_params([l_out_m_p])
all_grads_target_m_p = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target_m_p, all_params_target_m_p)]
all_grads_target_m_p = lasagne.updates.total_norm_constraint(all_grads_target_m_p, 10)
updates_target_m_p = adam(all_grads_target_m_p, all_params_target_m_p)

all_params_target_f_p = lasagne.layers.get_all_params([l_out_f_p])
all_grads_target_f_p = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target_f_p, all_params_target_f_p)]
all_grads_target_f_p = lasagne.updates.total_norm_constraint(all_grads_target_f_p, 10)
updates_target_f_p = adam(all_grads_target_f_p, all_params_target_f_p)

train_model_m_p = theano.function([x_sym, t_sym],
                              [loss_mean_target_m_p, output_m_p],
                              updates=updates_target_m_p)

test_model_m_p = theano.function([x_sym, t_sym],
                             [loss_mean_target_m_p, output_m_p])


train_model_f_p = theano.function([x_sym, t_sym],
                              [loss_mean_target_f_p, output_f_p],
                              updates=updates_target_f_p)

test_model_f_p = theano.function([x_sym, t_sym],
                               [loss_mean_target_f_p, output_f_p])

#####################################################################


num_min_batches = 100
n_batch = 100
epochs = 100

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x_a, batch_m_a, batch_f_a, batch_x_p, batch_m_p, batch_f_p = create_batches(n_batch)
        loss_train_m_a, out_m_a = train_model_m_a(batch_x_a, batch_m_a)
        loss_train_f_a, out_f_a = train_model_f_a(batch_x_a, batch_f_a)
        loss_train_m_p, out_m_p = train_model_m_p(batch_x_p, batch_m_p)
        loss_train_f_p, out_f_p = train_model_f_p(batch_x_p, batch_f_p)
        print 'loss male A %.10f' % loss_train_m_a,
        print 'loss female A %.10f' % loss_train_f_a,
        print 'loss male P %.10f' % loss_train_m_p,
        print 'loss female P %.10f' % loss_train_f_p
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(batch_m_p[0], interpolation='nearest', aspect='auto')
        plt.subplot(2, 2, 2)
        plt.imshow(batch_f_p[0], interpolation='nearest', aspect='auto')
        plt.subplot(2, 2, 3)
        plt.imshow(out_m_p[0], interpolation='nearest', aspect='auto')
        plt.subplot(2, 2, 4)
        plt.imshow(out_f_p[0], interpolation='nearest', aspect='auto')
        plt.show()

    batch_test_x_a, batch_test_m_a, batch_test_f_a, batch_test_x_p, batch_test_m_p, batch_test_f_p = create_batches(100, False)
    loss_test_m_a, out_m_a = test_model_m_a(batch_test_x_a, batch_test_m_a)
    loss_test_f_a, out_f_a = test_model_f_a(batch_test_x_a, batch_test_f_a)
    loss_test_m_p, out_m_p = test_model_m_p(batch_test_x_p, batch_test_m_p)
    loss_test_f_p, out_f_p = test_model_f_p(batch_test_x_p, batch_test_f_p)
    stop_time = time.time() - start_time

    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'loss male TEST A = %.10f \nloss female TEST A = %.10f' % (loss_test_m_a, loss_test_f_a),
    print 'loss male TEST P = %.10f \nloss female TEST P = %.10f' % (loss_test_m_p, loss_test_f_p)

# final test
test_x_a, test_m_a, test_f_a, test_x_p, test_m_p, test_f_p = create_batches(100, False)
l_m_a, out_m_a = test_model_m_a(test_x_a, test_m_a)
l_f_a, out_f_a = test_model_f_a(test_x_a, test_f_a)
l_m_p, out_m_p = test_model_m_p(test_x_p, test_m_p)
l_f_p, out_f_p = test_model_f_p(test_x_p, test_f_p)
out_out_m_a = out_m_a
out_out_f_a = out_f_a
out_out_m_p = out_m_p
out_out_f_p = out_f_p


savemat('RR_2models_polar_dense.mat', {'out_out_m_a': out_out_m_a,
                                       'out_out_f_a': out_out_f_p,
                                       'out_out_m_p': out_out_m_a,
                                       'out_out_f_p': out_out_f_p})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_out_m_a]),
         open('RR_2models_polar_dense_m_a'.format(date, NUM_UNITS_ENC), 'wb'))
pkl.dump(lasagne.layers.get_all_param_values([l_out_f_a]),
         open('RR_2models_polar_dense_f_a'.format(date, NUM_UNITS_ENC), 'wb'))
pkl.dump(lasagne.layers.get_all_param_values([l_out_m_p]),
         open('RR_2models_polar_dense_m_p'.format(date, NUM_UNITS_ENC), 'wb'))
pkl.dump(lasagne.layers.get_all_param_values([l_out_f_p]),
         open('RR_2models_polar_dense_f_p'.format(date, NUM_UNITS_ENC), 'wb'))


