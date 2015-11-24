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

theano.config.floatX = 'float64'

def pad_sequences(sequences, max_len, dtype='int32', padding='post', truncating='post', transpose=True, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[0])) * value).astype(dtype)
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

target_voice = loadmat('../data/single_voice_mocha.mat')['sel_wav_AS']
mixtures = loadmat('../data/single_voice_mocha.mat')['all_mix']
lengths = loadmat('../data/single_voice_mocha.mat')['LL']

# find max
n_features = target_voice.shape[0]

max_len = 0
for l in lengths:
    max_len = max(max_len, l[0])

data_x = mixtures[:, 0]
data_n = mixtures[:, 1]
data_x, mask_x = pad_sequences(data_x, max_len)
data_n, mask_n = pad_sequences(data_n, max_len)
# padding t
train_t = np.zeros((max_len, n_features))
mask_t = np.zeros((max_len, n_features))
train_t[:target_voice.shape[1], :] = target_voice.T
mask_t[:target_voice.shape[1], :] = 1.


# create teaching signal
n_batch = 12
n_test = 12

train_x = data_x[n_test:]
test_x = data_x[:n_test]
mask_train_x = mask_x[n_test:]
mask_test_x = mask_x[:n_test]

mask_n = np.zeros((len(data_n), max_len, n_features))
for i, d in enumerate(data_n):
    mask_n[i, :len(d), :] = 1.

train_n = data_n[n_test:]
test_n = data_n[:n_test]

mask_train_n = mask_n[n_test:]
mask_test_n = mask_n[:n_test]

print mask_n.shape
target_train = np.repeat(train_t[np.newaxis, :, :], n_batch, axis=0)
mask_target_train = np.repeat(mask_t[np.newaxis, :, :], n_batch, axis=0)

# print train_n.shape
# print train_t.shape
# print mask_x.shape
# print mask_n.shape
# print mask_t.shape
NUM_UNITS_ENC = 50
NUM_UNITS_DEC = 50
NUM_UNITS_BOT = 15

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()

l_in = lasagne.layers.InputLayer(shape=(n_batch, max_len, n_features))
l_mask = lasagne.layers.InputLayer(shape=(n_batch, max_len))

l_encoder = lasagne.layers.GRULayer(l_in,
                                    num_units=NUM_UNITS_ENC,
                                    mask_input=l_mask)

l_bottle = lasagne.layers.GRULayer(l_encoder,
                                   num_units=NUM_UNITS_BOT)

# From here I need to create two decoders
# For target
l_decoder_target = lasagne.layers.GRULayer(l_bottle, num_units=NUM_UNITS_DEC)
l_reshape_target = lasagne.layers.ReshapeLayer(l_decoder_target, (-1, NUM_UNITS_DEC))
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

# for noise
l_decoder_noise = lasagne.layers.GRULayer(l_bottle, num_units=NUM_UNITS_DEC)
l_reshape_noise = lasagne.layers.ReshapeLayer(l_decoder_noise, (-1, NUM_UNITS_DEC))
l_out_noise = lasagne.layers.DenseLayer(l_reshape_noise, num_units=n_features, nonlinearity=rectify)
l_out_reshape_noise = lasagne.layers.ReshapeLayer(l_out_noise, (-1, x_sym.shape[1], n_features))
output_noise = lasagne.layers.get_output(l_out_reshape_noise, inputs={l_in: x_sym, l_mask: mask_x_sym})

loss_all_noise = lasagne.objectives.squared_error(output_noise * mask_n_sym, n_sym)
loss_mean_noise = loss_all_noise.mean()
all_params_noise = lasagne.layers.get_all_params([l_out_reshape_noise])
all_grads_noise = [T.clip(g, -3, 3) for g in T.grad(loss_mean_noise, all_params_noise)]
all_grads_noise = lasagne.updates.total_norm_constraint(all_grads_noise, 3)

updates_noise = adam(all_grads_noise, all_params_noise)

train_noise = theano.function([x_sym, mask_x_sym, n_sym, mask_n_sym],
                              loss_mean_noise,
                              updates=updates_noise)

test_noise = theano.function([x_sym, mask_x_sym, n_sym, mask_n_sym],
                             [loss_mean_noise, output_noise])

# train_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym, n_sym, mask_n_sym],
#                               [train_target, train_noise])
#
# test_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym, n_sym, mask_n_sym],
#                              [test_target, test_noise])


num_min_batches = np.floor(len(train_x) / n_batch).astype('int32')
epochs = 50


for i in range(epochs):
    for j in range(num_min_batches):
        train_batch = train_x[j * n_batch: (j + 1) * n_batch]
        mask_batch = mask_train_x[j * n_batch: (j + 1) * n_batch]
        loss_train_target = train_target(train_batch,
                                         mask_batch,
                                         target_train,
                                         mask_target_train)
        loss_train_noise = train_noise(train_batch,
                                       mask_batch,
                                       train_n[j * n_batch: (j + 1) * n_batch],
                                       mask_train_n[j * n_batch: (j + 1) * n_batch])
    loss_test_target, _ = test_target(test_x, mask_test_x, target_train, mask_target_train)
    loss_test_noise, _ = test_noise(test_x, mask_test_x, test_n, mask_test_n)
    print ('-'*5 + ' epoch = %i ' + '-'*5) % i
    print 'loss_train_target = %.6f \nloss_test_target = %.6f' % (loss_train_target, loss_test_target)
    print 'loss_train_noise = %.6f \nloss_test_noise = %.6f' % (loss_train_noise, loss_test_noise)

# final test
loss_test_target, output_target = test_target(test_x, mask_test_x, target_train, mask_target_train)
loss_test_noise, output_noise = test_noise(test_x, mask_test_x, test_n, mask_test_n)
savemat('filtered_samples_multiple.mat', {'output_target': output_target, 'output_noise': output_noise})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_params(l_out_reshape_target),
         open('out_target_{}_{}_{}'.format(date, NUM_UNITS_ENC, NUM_UNITS_BOT), 'wb'))
pkl.dump(lasagne.layers.get_all_params(l_out_reshape_noise),
         open('out_noise_{}_{}_{}'.format(date, NUM_UNITS_ENC, NUM_UNITS_BOT), 'wb'))


