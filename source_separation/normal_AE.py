import lasagne
import theano
import theano.tensor as T
import numpy as np
import time as tm
from lasagne.nonlinearities import rectify,identity
from scipy.io import loadmat, savemat
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import init
from lasagne.updates import adam
from matplotlib import pyplot as plt

def load_mnist():

    data = np.load('../data/mnist.npz')
    num_classes = 10
    x_train, targets_train = data['X_train'].astype('float32'), data['y_train']
    x_valid, targets_valid = data['X_valid'].astype('float32'), data['y_valid']
    x_test, targets_test = data['X_test'].astype('float32'), data['y_test']


    def shared_dataset(x, y, borrow=True):
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    return shared_dataset(x_train, targets_train), shared_dataset(x_test, targets_test), shared_dataset(x_valid, targets_valid)



# from here we apply our data
cor = loadmat('../data/come_done_COR.mat')['COR']
rec_cor = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq, d))
n_epochs = 50
mask = np.array([1, 0])
mask2 = np.array([0, 1])
all_params = None

# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((512, d))
# create the model UGLY
l_in_1 = lasagne.layers.InputLayer(shape=(freq, d))
first_enc = lasagne.layers.DenseLayer(l_in_1, num_units=200, nonlinearity=rectify)
#sec_enc = lasagne.layers.DenseLayer(first_enc, num_units=2, nonlinearity=rectify)
#first_dec = lasagne.layers.DenseLayer(sec_enc, num_units=200, nonlinearity=rectify)
out = lasagne.layers.DenseLayer(first_enc, num_units=d, nonlinearity=identity)

x = T.dmatrix()
all_params = lasagne.layers.get_all_params(out)

output = lasagne.layers.get_output(out, inputs={l_in_1: x_sym})
loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()

loss_eval /= (2.*freq)


updates = lasagne.updates.adam(loss_eval, all_params)

train_model = theano.function([x_sym], loss_eval, updates=updates)

for t in range(time):
    x = cor[t, :, :]
    x_prime = np.zeros((d,1))
    for rate in range(rates/2):
        d11 = x[:, rate]
        d1 = np.outer(d11, d11)
        d22 = x[:, rates/2 + rate]
        d2 = np.outer(d22, d22)
        x_prime[rate*freq:(rate+1)*freq, 0] = np.real((d11 + d22) / 2)
        c_mat[:, slice(rate * freq, (rate + 1) * freq)] = np.real((d1 + d2) / 2)
    # c_mat = c_mat[:60, :]
    # x_prime = (x[:, :rates/2] + x[:, rates/2:]) / 2
    x_prime = x_prime.transpose()
    # c_mat is ready to be fed and train the VAE the whole training set has 'freq' #samples
    shared_c_mat = theano.shared(np.asarray(c_mat, dtype=theano.config.floatX), borrow=True)
    shared_x_prime = theano.shared(np.asarray(x_prime, dtype=theano.config.floatX), borrow=True)

    # print c_mat.shape

    for epoch in range(n_epochs):
        eval_train = train_model(c_mat)
        print " %.10f (epoch=%i)" % (eval_train, epoch)

    c_recon = lasagne.layers.get_output(out, inputs={l_in_1: x_sym}).eval({x_sym: c_mat})
    # x_rec_no_mask = lasagne.layers.get_output(sec_enc, inputs={l_in_1: x_sym}).eval({x_sym: x_prime})
    # x_masked_1 = x_rec_no_mask * mask
    # x_masked_2 = x_rec_no_mask * mask2
    # helf_way_1 = lasagne.layers.get
    #x_recon = lasagne.layers.get_output(out, inputs={l_in_1: x_sym}).eval({x_sym: x_prime})

    plt.figure
    plt.subplot(1, 2, 1)
    plt.imshow(c_mat, interpolation='nearest', aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(c_recon, interpolation='nearest', aspect='auto')
    plt.show()
    # tx = np.arange(d)
    # plt.plot(tx, x_prime.reshape((d,)), 'b', tx, x_recon.reshape((d,)), 'r')
    # plt.show()
savemat('come_done_filtered.mat', {'rec_cor': rec_cor})