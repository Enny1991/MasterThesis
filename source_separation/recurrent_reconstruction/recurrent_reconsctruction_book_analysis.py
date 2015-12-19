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


data_m = loadmat('../../data/m_f_book_24.mat')['hunchback']
data_f = loadmat('../../data/m_f_book_24.mat')['screw']
data_mix = loadmat('../../data/m_f_book_24.mat')['hunchback_screw']

data_mix /= np.max(np.abs(data_mix))
data_f /= np.max(np.abs(data_f))
data_m /= np.max(np.abs(data_m))
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

NUM_UNITS_ENC = 200
NUM_UNITS_DEC = 200

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()
rng = np.random.RandomState(123)
initial_W = np.asarray(
            rng.uniform(
                    low=-4 * np.sqrt(6. / (NUM_UNITS_ENC + NUM_UNITS_ENC*2)),
                    high=4 * np.sqrt(6. / (NUM_UNITS_ENC + NUM_UNITS_ENC*2)),
                    size=(NUM_UNITS_ENC,  NUM_UNITS_ENC*2)
            ),
            dtype=theano.config.floatX
        )
initial_W2 = np.asarray(
            rng.uniform(
                    low=-4 * np.sqrt(6. / (NUM_UNITS_ENC*2 + n_features*2)),
                    high=4 * np.sqrt(6. / (NUM_UNITS_ENC*2 + n_features*2)),
                    size=(NUM_UNITS_ENC*2,  n_features*2)
            ),
            dtype=theano.config.floatX
        )

W = theano.shared(value=initial_W, name='W', borrow=True)
        # # self.W_y_kappa = theano.shared(value=initial_W, name='W_y_kappa', borrow=True)
b = theano.shared(
                value=np.zeros(
                    NUM_UNITS_ENC*2,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
W2 = theano.shared(value=initial_W2, name='W', borrow=True)
        # # self.W_y_kappa = theano.shared(value=initial_W, name='W_y_kappa', borrow=True)
b2 = theano.shared(
                value=np.zeros(
                    n_features*2,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

print lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)
#l_dec_bwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=True)

#l_last_hid_dec = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)
# l_encoder_2 = lasagne.layers.GRULayer(l_encoder_1,
#                                     num_units=NUM_UNITS_ENC,
#                                     mask_input=l_mask)

# l_decoder = LSTMAttentionDecodeFeedbackLayer(l_encoder_1,
#                                    num_units=NUM_UNITS_DEC,
#                                               aln_num_units=100,
#                                         n_decodesteps=win_len)
l_decoder = lasagne.layers.GRULayer(l_dec_fwd, num_units=n_features*2)
# print lasagne.layers.get_output(l_decoder, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape
#
# l_reshape = lasagne.layers.ReshapeLayer(l_decoder, (-1, NUM_UNITS_DEC))
# print lasagne.layers.get_output(l_reshape, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape
#
# l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=n_features * 2, nonlinearity=rectify)
# print lasagne.layers.get_output(l_dense, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape
# l_out = lasagne.layers.ReshapeLayer(l_dense, (-1, max_len, n_features * 2))
#
# print lasagne.layers.get_output(l_out, inputs={l_in: x_sym}).eval({x_sym: ftrain_x}).shape


output = lasagne.layers.get_output(l_decoder, inputs={l_in: x_sym})

# output = T.nnet.sigmoid(T.dot(output, W) + b)
# output = T.nnet.sigmoid(T.dot(output, W2) + b2)
# print lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym}).eval({x_sym:test_x,mask_x_sym:mask_test_x}).shape

loss_all_target = lasagne.objectives.squared_error(output, t_sym).sum()

loss_mean_target = loss_all_target / n_batch

# print loss_mean_target.eval({x_sym:test_x,mask_x_sym:mask_test_x, t_sym: target_train, mask_t_sym: mask_target_train})

all_params_target = lasagne.layers.get_all_params([l_decoder])
all_grads_target = [T.clip(g, -10, 10) for g in T.grad(loss_mean_target, all_params_target)]
all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 10)
updates_target = adam(all_grads_target, all_params_target)

train_model = theano.function([x_sym, t_sym],
                              [loss_mean_target, output],
                              updates=updates_target)

test_model = theano.function([x_sym, t_sym],
                             [loss_mean_target, output])


data = pkl.load(open('recurrent_reconstruction_PCA_14:49_07:12:2015_200_lonely_24', 'r'))

# lasagne.layers.set_all_param_values([l_decoder], data)
test_x, test_t = create_batches(100, False)
out = lasagne.layers.get_output(l_dec_fwd, inputs={l_in: x_sym}).eval({x_sym: test_x})
savemat('out_only_mix_24_b.mat', {'mix_b' :out})

