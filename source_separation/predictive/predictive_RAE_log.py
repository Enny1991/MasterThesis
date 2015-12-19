from __future__ import division
import lasagne
import numpy as np
import time
from lasagne.layers import GRULayer, LSTMLayer, DenseLayer, InputLayer, ReshapeLayer, SliceLayer, Layer, DropoutLayer, ConcatLayer
import theano
from lasagne import init
import theano.tensor as T
from scipy.io import loadmat, savemat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import math
from lasagne.updates import adam, sgd, rmsprop
from lasagne.nonlinearities import rectify
import matplotlib.pyplot as plt
from scipy.special import gamma


def make_shared(x, borrow=True):
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
        return shared_x


def pad_sequences(sequences, max_len, dtype='float32', padding='post', truncating='post', transpose=True, value=0.):
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

c = - 0.5 * math.log(2*math.pi)
data = loadmat('../../data/all_mocha_wavs.mat')['FEM_AUD']
data = data[0]
print len(data)
print data[0].shape


#

n_features = data[0].shape[1]
n_features_model = 30
total_samples = len(data)
train_samples = 430

print 'n_features %i' % n_features
max_len = 0
for l in data:
    max_len = max(max_len, l.shape[0])

print 'max_len: %i' % max_len
data_x, mask_x = pad_sequences(data, max_len, transpose=False)
mask_out = np.zeros((len(data), max_len, n_features_model))
for i, d in enumerate(data):
    mask_out[i, :len(d), :] = 1.

# here a need to create the data which is divide the data set in 6 datasets
# the frequencies are low-medium-high with the indices being 0->127
# hardcoded the ranges are:
# 0-30 / 20-50 / 40-70 / 60-90 / 80-110 / 100-128
slices = [slice(0, 30), slice(20, 50), slice(40, 70),  slice(60, 90), slice(80, 110), slice(100, 128)]
overlap = [slice(20, 30), slice(40, 50), slice(60, 70), slice(80, 90), slice(100, 110)]
len_slices = [30, 30, 30, 30, 30, 28]

# norm
# for i, sample in enumerate(data_x):
#     a = np.max(sample)
#     data_x[i] = sample / a
#     print a
# for i, sample in enumerate(data_x):
#     a = np.max(sample)
#     print a

#

# data_x is in the shape (n_samples, max_len, features)
# TODO normalize the data before

DATA_X = np.zeros((len(data_x), len(slices), max_len, n_features_model))
DATA_T = np.zeros_like(DATA_X)
for i, sample in enumerate(data_x):
    for j, sl in enumerate(slices):
        DATA_X[i, j, :, :len_slices[j]] = sample[:, sl]
        DATA_T[i, j, :-1, :len_slices[j]] = sample[1:, sl]  # shifted version for target
print 'DATA_X shape: {}'.format(DATA_X.shape)
# to check
# savemat('DATA_X.mat', {'data_x': DATA_X,'data_t': DATA_T})
# the mask is the same for all the 6 slices
# mask_out has already been calculated and its the same for all slices but now it is 30 features
# separation



train_x = DATA_X[:train_samples]
train_mask_x = mask_x[:train_samples]
train_mask_out = mask_out[:train_samples]
train_t = DATA_T[:train_samples]
test_x = DATA_X[train_samples:]
test_mask_x = mask_x[train_samples:]
test_mask_out = mask_out[train_samples:]
test_t = DATA_T[train_samples:]

print '-'*10 + 'SIZES' + '-'*10
print'train_x: {}'.format(train_x.shape)
print'train_mask_x: {}'.format(train_mask_x.shape)
print'train_mask_out: {}'.format(train_mask_out.shape)
print'train_t: {}'.format(train_t.shape)
print'test_x: {}'.format(test_x.shape)
print'test_mask_x: {}'.format(test_mask_x.shape)
print'test_mask_out: {}'.format(test_mask_out.shape)
print'test_t: {}'.format(test_t.shape)
print '-'*25

# make dataset shared
train_x = make_shared(train_x)
train_mask_x = make_shared(train_mask_x)
train_mask_out = make_shared(train_mask_out)
train_t = make_shared(train_t)
test_x = make_shared(test_x)
test_mask_x = make_shared(test_mask_x)
test_mask_out = make_shared(test_mask_out)
test_t = make_shared(test_t)

# Now i will create a class that comprise only one RNN I will afterward take 6 istances

class PRAE:
    def __init__(self, num_batch, max_len, n_features, hidden=[200, 200], **kwargs):
        self.num_batch = num_batch
        self.n_features = n_features
        self.max_len = max_len
        self.hidden = hidden
        rng = np.random.RandomState(123)
        self.drng = rng
        self.rng = RandomStreams(rng.randint(2 ** 30))

        # params
        initial_W = np.asarray(
            rng.uniform(
                    low=-4 * np.sqrt(6. / (self.hidden[1] + self.n_features)),
                    high=4 * np.sqrt(6. / (self.hidden[1] + self.n_features)),
                    size=(self.hidden[1], self.n_features)
            ),
            dtype=theano.config.floatX
        )

        self.W = theano.shared(value=initial_W, name='W', borrow=True)
        # # self.W_y_kappa = theano.shared(value=initial_W, name='W_y_kappa', borrow=True)
        self.b = theano.shared(
                value=np.zeros(
                    self.n_features,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        # self.b_y_kappa = theano.shared(
        #         value=np.zeros(
        #             self.n_features,
        #             dtype=theano.config.floatX
        #         ),
        #         name='b',
        #         borrow=True
        #     )


        # I could directly create the model here since it is fixed
        self.l_in = InputLayer(shape=(None, self.max_len, self.n_features))
        self.mask_input = InputLayer(shape=(None, self.max_len))
        first_hidden = GRULayer(self.l_in, mask_input=self.mask_input, num_units=hidden[0])
        # l_shp = ReshapeLayer(first_hidden, (-1, hidden[0]))
        # l_dense = DenseLayer(l_shp, num_units=self.hidden[0], nonlinearity=rectify)
        # l_drop = DropoutLayer(l_dense, p=0.5)
        # l_shp = ReshapeLayer(l_drop, (-1, self.max_len, self.hidden[0]))
        self.model = GRULayer(first_hidden, num_units=hidden[1])
        # self.model = ConcatLayer([first_hidden, second_hidden], axis=2)
        # l_shp = ReshapeLayer(second_hidden, (-1, hidden[1]))
        # l_dense = DenseLayer(l_shp, num_units=self.n_features, nonlinearity=rectify)
        # To reshape back to our original shape, we can use the symbolic shape
        # variables we retrieved above.
        #self.model = ReshapeLayer(l_dense, (-1, self.max_len, self.n_features))
        # if now I put a dense layer this will collect all the output temporally which is what I want, I'll have to fix
        # the dimensions probably later
        # For every gaussian in the sum I need 3 values plus a value for the total scale
        # the output of this layer will be (num_batch, num_units, max_len) TODO check size

    def get_output_shape_for(self):
        return self.model.get_output_shape_for(self.num_batch, self.max_len, self.hidden[1])

    def get_output_y(self, x):
        return T.nnet.relu(T.dot(x, self.W) + self.b)


    def build_model(self, train_x, train_mask_x, train_mask_out, train_target,
                    test_x, test_mask_x, test_mask_out, test_target):
        self.train_x = train_x
        self.train_mask_x = train_mask_x
        self.train_mask_out = train_mask_out
        self.train_target = train_target
        self.test_x = test_x
        self.test_mask_x = test_mask_x
        self.test_mask_out = test_mask_out
        self.test_target = test_target
        self.index = T.iscalar('index')
        self.num_batch_test = T.iscalar('index')
        self.b_slice = slice(self.index * self.num_batch, (self.index + 1) * self.num_batch)

        sym_x = T.dtensor3()
        sym_mask_x = T.dmatrix()
        sym_target = T.dtensor3()
        sym_mask_out = T.dtensor3()
        # sym_mask_out = T.dtensor3() should not be useful since output is still zero
        # TODO think about this if it is true

        out = lasagne.layers.get_output(self.model, inputs={self.l_in: sym_x, self.mask_input: sym_mask_x})
        out_out = self.get_output_y(out)
        loss = T.mean(lasagne.objectives.squared_error(out_out, sym_target)) / self.num_batch

        out_test = lasagne.layers.get_output(self.model, inputs={self.l_in: sym_x, self.mask_input: sym_mask_x})
        out_out_test = self.get_output_y(out_test)
        loss_test = T.mean(lasagne.objectives.squared_error(out_out_test, sym_target)) / self.num_batch_test

        all_params = [self.W] + [self.b] +lasagne.layers.get_all_params(self.model)
        all_grads_target = [T.clip(g, -3, 3) for g in T.grad(loss, all_params)]
        all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 3)
        updates_target = adam(all_grads_target, all_params)

        train_model = theano.function([self.index],
                                      [loss, out_out],
                                      givens={sym_x: self.train_x[self.b_slice],
                                              sym_mask_x: self.train_mask_x[self.b_slice],
                                              sym_target: self.train_target[self.b_slice],
                                              },
                                      updates=updates_target)
        test_model = theano.function([self.num_batch_test],
                                     [loss_test, out_out_test],
                                     givens={sym_x: self.test_x,
                                             sym_mask_x: self.test_mask_x,
                                             sym_target: self.test_target,
                                             })

        return train_model, test_model


###### Run the model ######
num_batches = 5
n_batch = int((train_samples / num_batches))
n_epochs = 10
n_batch_test = total_samples - train_samples


MODEL = [PRAE(n_batch, max_len, n_features_model, hidden=[50, 50]) for _ in range(6)]

TRAINER = []
TESTER = []
for i, model in enumerate(MODEL):
    p, q = model.build_model(train_x[:, i, :, :],
                             train_mask_x,
                             train_mask_out,
                             train_t[:, i, :, :],
                             test_x[:, i, :, :],
                             test_mask_x,
                             test_mask_out,
                             test_t[:, i, :, :],
                             )
    TRAINER += [p]
    TESTER += [q]

ex = np.zeros((6, total_samples - train_samples, max_len, n_features_model))

for e in range(n_epochs):
    for j, trainer in enumerate(TRAINER):
        for k in range(num_batches):
            loss, theta = trainer(k)
            print'epoch=%i/%i' % ((e + 1), n_epochs) + '-'*1,
            print'model=%i/6' % (j + 1) + '-'*1,
            print'batch=%i/%i' % (k+1, num_batches) + '-'*3
            print'loss = %.10f' % loss
            print'-'*25

    print'-'*10 + 'TESTING' + '-'*10
    for j, tester in enumerate(TESTER):
        loss, theta = tester(total_samples - train_samples)
        # savemat('log.mat', {'theta': theta})
        ex[j, :, :, :] = theta
        print'loss model %i = %.10f' % (j+1, loss)
    print '-'*25

    # print some shit to see the output
    # merge output
    merged = np.zeros((total_samples - train_samples, max_len, n_features))
    for i in range(total_samples - train_samples):
        for l, sl in enumerate(slices):
            merged[i, :, sl] += ex[l, i, :, :len_slices[l]]
    for i in overlap:
        merged[:, :, i] /= 2
    sel = 0
    sel_0 = sel + train_samples
    original = data_x[sel_0]
    fig, ax = plt.subplots(nrows=1, ncols=10)
    fig.tight_layout()
    plt.subplot2grid((1, 10), (0, 0), colspan=4)
    plt.imshow(original, interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 4), colspan=1)
    plt.imshow(ex[0, sel], interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 5), colspan=1)
    plt.imshow(ex[1, sel], interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 6), colspan=1)
    plt.imshow(ex[2, sel], interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 7), colspan=1)
    plt.imshow(ex[3, sel], interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 8), colspan=1)
    plt.imshow(ex[4, sel], interpolation='nearest', aspect='auto')
    plt.subplot2grid((1, 10), (0, 9), colspan=1)
    plt.imshow(ex[5, sel], interpolation='nearest', aspect='auto')
    plt.show()
    plt.figure()
    plt.imshow(merged[sel], interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.show()

savemat('out_predictive_rec.mat', {'merged': merged})





