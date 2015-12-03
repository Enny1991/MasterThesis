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
from decoder_attention import LSTMAttentionDecodeFeedbackLayer
theano.config.floatX = 'float64'

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def create_batches(sample, mask, win_len):
    n = sample.shape[0] / win_len
    train_x = np.zeros((n, win_len, sample.shape[1]))
    mask_x = np.zeros((n, win_len))
    mask_y = np.zeros((n, win_len, sample.shape[1]))
    for i in range(n):
        train_x[i] = sample[i*win_len: (i+1) * win_len]
        mask_x[i] = mask[i*win_len: (i+1) * win_len]
        idx = indices(mask_x[i], lambda x: x == 1)
        mask_y[i, idx, :] = 1.

    return train_x, mask_x, mask_y

def pad_sequences(sequences, max_len, dtype='int32', padding='post', truncating='post', transpose=True, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[1])) * value).astype(dtype)
    mask = (np.zeros((nb_samples, max_len))).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[:, -max_len:]
        elif truncating == 'post':
            trunc = s[:, :max_len]
        if transpose:
            trunc = trunc.T
        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
            mask[idx, :len(trunc)] = 1.
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
            mask[idx, -len(trunc):] = 1.
    return x, mask

data = loadmat('../data/all_mocha_wavs.mat')['FEM_AUD']
data = data[0]
print len(data)
print data[0].shape

# find max

n_features = data[0].shape[1]

print 'n_features %i' % n_features
win_len = 20
max_len = 0
for l in data:
    max_len = max(max_len, l.shape[0])
max_len += win_len - max_len % win_len

print max_len
data_x, mask_x = pad_sequences(data, max_len, transpose=False)
mask_out = np.zeros((len(data), max_len, n_features))
for i, d in enumerate(data):
    mask_out[i, :len(d), :] = 1.

# create teaching signal

n_batch = 40
n_test = 60

train_x = data_x[n_test:]
test_x = data_x[:n_test]
mask_train_x = mask_x[n_test:]
mask_test_x = mask_x[:n_test]
mask_out_train = mask_out[n_test:]
mask_out_test = mask_out[:n_test]
NUM_UNITS_ENC = 200
NUM_UNITS_DEC = 200

ftrain_x, fmask_x, fmask_y = create_batches(train_x[0], mask_train_x[0], win_len)
print ftrain_x.shape
print fmask_x.shape
print fmask_y.shape

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()

l_in = lasagne.layers.InputLayer(shape=(n_batch, win_len, n_features))
l_mask = lasagne.layers.InputLayer(shape=(n_batch, win_len))

l_encoder_1 = lasagne.layers.GRULayer(l_in,
                                    num_units=NUM_UNITS_ENC,
                                    mask_input=l_mask)
# l_encoder_2 = lasagne.layers.GRULayer(l_encoder_1,
#                                     num_units=NUM_UNITS_ENC,
#                                     mask_input=l_mask)

# l_decoder = LSTMAttentionDecodeFeedbackLayer(l_encoder_1,
#                                    num_units=NUM_UNITS_DEC,
#                                               aln_num_units=100,
#                                         n_decodesteps=win_len)
l_decoder = lasagne.layers.GRULayer(l_encoder_1,
                                    num_units=NUM_UNITS_DEC,
                                    mask_input=l_mask)

l_reshape_target = lasagne.layers.ReshapeLayer(l_decoder, (-1, NUM_UNITS_DEC))
l_out_target = lasagne.layers.DenseLayer(l_reshape_target, num_units=n_features, nonlinearity=rectify)
l_out_reshape_target = lasagne.layers.ReshapeLayer(l_out_target, (-1, x_sym.shape[1], n_features))
output_target = lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym})

# print lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym}).eval({x_sym:test_x,mask_x_sym:mask_test_x}).shape

loss_all_target = lasagne.objectives.squared_error(output_target * mask_t_sym, t_sym)

loss_mean_target = loss_all_target.mean()

# print loss_mean_target.eval({x_sym:test_x,mask_x_sym:mask_test_x, t_sym: target_train, mask_t_sym: mask_target_train})

all_params_target = lasagne.layers.get_all_params([l_out_reshape_target])
all_grads_target = [T.clip(g, -3, 3) for g in T.grad(loss_mean_target, all_params_target)]
all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 3)
updates_target = adam(all_grads_target, all_params_target)

train_target = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                               loss_mean_target,
                               updates=updates_target)

test_target = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                              [loss_mean_target, output_target])


num_min_batches = np.floor(len(train_x) / n_batch).astype('int32')
epochs = 50
sent = 400

for i in range(epochs):
    start_time = time.time()
    for j in range(num_min_batches):
        train_batch = train_x[j * n_batch: (j + 1) * n_batch]
        mask_batch = mask_train_x[j * n_batch: (j + 1) * n_batch]
        mask_out_batch = mask_out_train[j * n_batch: (j + 1) * n_batch]
        for k in range(n_batch):
            t, m, m_o = create_batches(train_batch[k], mask_batch[k], win_len)
            loss_train_target = train_target(t, m, t, m_o)

    loss_test = 0
    for j in range(len(test_x)):
        t, m, m_o = create_batches(test_x[k], mask_test_x[k], win_len)
        l, _ = test_target(t, m, t, m_o)
        loss_test += l
    loss_test /= len(test_x)
    stop_time = time.time() - start_time
    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'loss_train_target = %.6f \nloss_test_target = %.6f' % (loss_train_target, loss_test)

# final test
out_out = np.zeros((len(test_x), max_len, n_features))
loss_test = 0
for j in range(len(test_x)):
    t, m, m_o = create_batches(test_x[j], mask_test_x[j], win_len)
    l, out = test_target(t, m, t, m_o)
    out_out[j, :, :] = np.reshape(out, (max_len, n_features))
    loss_test += l
loss_test /= len(test_x)
print 'TEST = %.6f' % (loss_test)

savemat('recurrent_reconstruction_200.mat', {'out_out': out_out})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_params(l_out_reshape_target),
         open('recurrent_reconstruction_{}_{}'.format(date, NUM_UNITS_ENC), 'wb'))


