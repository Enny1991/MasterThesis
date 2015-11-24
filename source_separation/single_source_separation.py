import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import matlab.engine

eng = matlab.engine.start_matlab()


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
target_train = np.repeat(train_t[np.newaxis, :, :], n_batch, axis=0)
mask_target_train = np.repeat(mask_t[np.newaxis, :, :], n_batch, axis=0)

# print train_n.shape
# print train_t.shape
# print mask_x.shape
# print mask_n.shape
# print mask_t.shape
NUM_UNITS_ENC = 100
NUM_UNITS_DEC = 100
NUM_UNITS_BOT = 30



x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_t_sym = T.dtensor3()

l_in = lasagne.layers.InputLayer(shape=(n_batch, max_len, n_features))
l_mask = lasagne.layers.InputLayer(shape=(n_batch, max_len))

l_encoder = lasagne.layers.GRULayer(l_in,
                                    num_units=NUM_UNITS_ENC,
                                    mask_input=l_mask)

l_bottle = lasagne.layers.GRULayer(l_encoder,
                                   num_units=NUM_UNITS_BOT)

# Let's try before just to train one decoder to extract the target
# TODO add to decoders so one extracts the noise too
l_decoder = lasagne.layers.GRULayer(l_bottle,
                                    num_units=NUM_UNITS_DEC)

l_reshape = lasagne.layers.ReshapeLayer(l_decoder, (-1, NUM_UNITS_DEC))

l_out = lasagne.layers.DenseLayer(l_reshape, num_units=n_features, nonlinearity=rectify)

l_out_reshape = lasagne.layers.ReshapeLayer(l_out, (-1, x_sym.shape[1], n_features))


output = lasagne.layers.get_output(l_out_reshape, inputs={l_in: x_sym, l_mask: mask_x_sym})


loss_all = lasagne.objectives.squared_error(output * mask_t_sym, t_sym)

loss_mean = loss_all.mean()

all_params = lasagne.layers.get_all_params([l_out_reshape])
all_grads = [T.clip(g, -3, 3) for g in T.grad(loss_mean, all_params)]
all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)

updates = adam(all_grads, all_params)

train_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                              loss_mean,
                              updates=updates)

test_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                             [loss_mean, output])


num_min_batches = np.floor(len(train_x) / n_batch).astype('int32')
epochs = 50


for i in range(epochs):
    for j in range(num_min_batches):
        loss_train = train_model(train_x[j * n_batch: (j + 1) * n_batch],
                                 mask_train_x[j * n_batch: (j + 1) * n_batch],
                                 target_train,
                                 mask_target_train)
    loss_test, _ = test_model(test_x, mask_test_x, target_train, mask_target_train)
    print ('-'*5 + ' epoch = %i ' + '-'*5) % i
    print 'loss_train = %.6f \nloss_test = %.6f' % (loss_train, loss_test)

# final test
_, output = test_model(test_x, mask_test_x, target_train, mask_target_train)

savemat('filtered_samples.mat', {'output': output})

