from __future__ import division
import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
import cPickle as pkl
import matlab.engine

# print 'Loading MATLAB...'
# eng = matlab.engine.start_matlab()
# eng.load_wavs_mocha(nargout=0)
# theano.config.floatX = 'float64'


def pad_sequences(sequences, max_len, dtype='float32', padding='post', truncating='post', transpose=False, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[1])) * value).astype(dtype)
    mask = (np.zeros((nb_samples, max_len))).astype('int32')
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


def create_batches(sample, mask, win_len):
    n = sample.shape[0] - win_len
    train_x = np.zeros((n, win_len, sample.shape[1]))
    mask_x = np.zeros((n, win_len))
    train_y = np.zeros((n, sample.shape[1]))
    mask_y = np.zeros((n, sample.shape[1]))
    for i in range(n):
        train_x[i] = sample[i: i + win_len]
        mask_x[i] = mask[i: i + win_len]
        train_y[i] = sample[i + win_len]
        mask_y[i] = mask[i + win_len]

    return train_x, mask_x, train_y, mask_y

# data
data = loadmat('../data/all_mocha_wavs.mat')['FEM_AUD']
data = data.reshape((460, ))
win_len = 50  # TODO CV this shit
max_len = 0
for d in data:
    max_len = max(max_len, len(d))
max_len += win_len - max_len % 50
print max_len
data, mask = pad_sequences(data, max_len)

train = data[:430]  # 430
mask_train = mask[:430]
test = data[430:]  # 30
mask_test = mask[:430]

print data.shape

NUM_UNITS_ENC = 400
NUM_UNITS_HID = 200
n_features = data.shape[2]
NUM_OUT = n_features

n_batch = max_len - win_len

# check input
print train.shape
print mask_train.shape
print test.shape
print mask_test.shape

ftrain_x, fmask_x, ftrain_y, fmask_y = create_batches(train[0],mask_train[0], win_len)
print ftrain_x.shape
print fmask_x.shape
print ftrain_y.shape
print fmask_y.shape
savemat('test_shape.mat',{'train_x': ftrain_x, 'mask_x': fmask_x, 'train_y': ftrain_y, 'mask_y':fmask_y})

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dmatrix()
mask_t_sym = T.dmatrix()

print 'Creating Model...'

l_in = lasagne.layers.InputLayer(shape=(n_batch, win_len, n_features))
l_mask = lasagne.layers.InputLayer(shape=(n_batch, win_len))

l_encoder = lasagne.layers.GRULayer(l_in,
                                    num_units=NUM_UNITS_ENC,
                                    mask_input=l_mask)

l_slice = lasagne.layers.SliceLayer(l_encoder, indices=-1, axis=1)

l_hid = lasagne.layers.DenseLayer(l_slice, num_units=NUM_UNITS_HID, nonlinearity=rectify)

l_out = lasagne.layers.DenseLayer(l_hid, num_units=NUM_OUT, nonlinearity=identity)


output = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask: mask_x_sym})

# print lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym}).eval({x_sym:test_x,mask_x_sym:mask_test_x}).shape

loss_all_target = lasagne.objectives.squared_error(output * mask_t_sym, t_sym)

loss_mean_target = loss_all_target.mean()

# print loss_mean_target.eval({x_sym:test_x,mask_x_sym:mask_test_x, t_sym: target_train, mask_t_sym: mask_target_train})

all_params_target = lasagne.layers.get_all_params([l_out])
all_grads_target = [T.clip(g, -3, 3) for g in T.grad(loss_mean_target, all_params_target)]
all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 3)
updates_target = adam(all_grads_target, all_params_target)

train_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                              loss_mean_target,
                              updates=updates_target)

test_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                             [loss_mean_target, output])

n_sent = 250

epochs = 25
print 'Training...'
for i in range(epochs):
    print ('-'*5 + ' epoch = %i ' + '-'*5) % i
    for k in range(n_sent):

        [train_x, mask_x, train_y, mask_y] = create_batches(train[k], mask_train[k], win_len)
        loss_train_target = train_model(train_x,
                                        mask_x,
                                        train_y,
                                        mask_y)
        print 'loss_sample %i = %.6f ' % (k, loss_train_target)
    mean_loss = 0
    for k in range(5):
        [test_x, mask_x, test_y, mask_y] = create_batches(test[k], mask_test[k], win_len)
        loss_test, _ = test_model(test_x, mask_x, test_y, mask_y)
        mean_loss += loss_test
    m_l = mean_loss / 5

    print 'mean loss_test = %.6f ' % m_l

# final test
output = np.zeros((30, n_batch, n_features))
mean_loss = 0
for k in range(30):
    [test_x, mask_x, test_y, mask_y] = create_batches(test[k], mask_test[k], win_len)
    loss, out = test_model(test_x, mask_x, test_y, mask_y)
    mean_loss += loss
    output[k] = out
    m_l = mean_loss / 30
print 'FINAL TEST: mean loss_test = %.6f ' % m_l

savemat('simple_rec.mat', {'output': output})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_params(l_out),
         open('predictive_GRU_{}_{}_{}'.format(date, NUM_UNITS_ENC, NUM_UNITS_HID), 'wb'))

# eng.exit()


