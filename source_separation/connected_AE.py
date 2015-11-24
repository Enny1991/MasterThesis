
import lasagne
import theano
import theano.tensor as T
import numpy as np
import time as tm
from lasagne.nonlinearities import rectify, identity, tanh
from scipy.io import loadmat, savemat
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import init
from lasagne.updates import adam
import matplotlib.pyplot as plt
from lasagne.regularization import l2
import copy


def createMLP(layers,s):
    l_in = lasagne.layers.InputLayer(shape=(None, s))
    prev_layer = l_in
    Ws = []
    for layer in layers:
        enc = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify, W=init.Uniform(0.001))
        Ws += [enc.W]
        drop = lasagne.layers.DropoutLayer(enc, p=0.5)
        prev_layer = drop
    idx = 1
    last_enc = prev_layer
    for layer in layers[-2::-1]:
        print layer
        dec = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify, W=Ws[-idx].T)
        idx += 1
        drop = lasagne.layers.DropoutLayer(dec, p=0.0)
        prev_layer = drop
    model = lasagne.layers.DenseLayer(prev_layer, num_units=s, nonlinearity=identity, W=Ws[0].T)

    x_sym = T.dmatrix()
    all_params = lasagne.layers.get_all_params(model)
    output = lasagne.layers.get_output(model, inputs={l_in: x_sym})
    loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()
    loss_eval /= (2.*batch_size)
    updates = lasagne.updates.adam(loss_eval, all_params)

    return l_in, model, last_enc ,theano.function([x_sym], loss_eval, updates=updates)









####

rng = np.random.RandomState(42)
cor = loadmat('../data/come_done_COR_5R.mat')['COR']
rec_cor_1 = np.zeros_like(cor)
rec_cor_2 = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq, d))
mask = np.zeros((freq+1, 2))
mask2 = np.zeros((freq+1, 2))
mask[:, 0] = 1.
mask[:, 1] = 1.
all_params = None
a2 = 512
# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((64, d))
x_fake_2 = np.ones((32, d))


n_epochs = 50
batch_size = 500
layers = [200, 2]
Ws = []
collect_activity = np.zeros((time, rates / 2, 200))

l_in, model, last_enc, train_model = createMLP(layers, a2)
fig, axes = plt.subplots(nrows=2, ncols=3)

for t in range(time):
    x = cor[t, :, :]
    x_prime = np.zeros((rates/2, freq))
    for rate in range(rates/2):
        d11 = x[:, rate]
        d1 = np.outer(d11, d11)
        d22 = x[:, rates/2 + rate]
        d2 = np.outer(d22, d22)
        x_prime[rate, :] = np.real((d11 + d22) / 2)
        c_mat[:, slice(rate * freq, (rate + 1) * freq)] = np.real((d1 + d2) / 2)
    # x_prime = x_prime.transpose()
    # c_mat[-1, :] = x_prime
    c_mat_lim = c_mat.transpose()
    # c_mat_lim = (c_mat_lim - np.min(c_mat_lim)) / (np.max(c_mat_lim) - np.min(c_mat_lim))
    # c_mat_lim = c_mat_lim * 2 - 1
    for epoch in range(n_epochs):
        # c_mat_sc = c_mat[np.random.randint(0, freq+1, size=(freq + 1))]
        # c_mat_sc += rng.rand(c_mat_sc.shape[0], c_mat_sc.shape[1]) / (np.max(c_mat_sc) * 10)
        # for i in range(8):
        eval_train = train_model(c_mat_lim)
    print " %.10f (time=%i)" % (eval_train, t)

    c_recon = lasagne.layers.get_output(model, {l_in: x_sym}, deterministic=True).eval({x_sym: c_mat_lim})
    middle_act = lasagne.layers.get_output(last_enc, {l_in: x_sym}, deterministic=True).eval({x_sym: x_prime})
    x_recon = lasagne.layers.get_output(model, {l_in: x_sym}, deterministic=True).eval({x_sym: x_prime})

    # high = np.zeros_like(middle_act)
    # mod_mid = copy.deepcopy(middle_act)
    # for row, i in zip(mod_mid, range(len(middle_act))):
    #     d = np.argsort(row)
    #     d_min = d[:len(d)/2]
    #     d_max = d[len(d)/2:]
    #     row[d_min] = 0
    #     row[d_max] = 1
    #     high[i, :] = row
    # collect_activity[t, :, :] = middle_act


    plt.subplot2grid((2, 3), (0, 0))
    plt.imshow(c_mat_lim, interpolation='nearest', aspect='auto')
    plt.subplot2grid((2, 3), (0, 1))
    plt.imshow(c_recon, interpolation='nearest', aspect='auto')
    plt.subplot2grid((2, 3), (1, 0))
    plt.imshow(x_prime, interpolation='nearest', aspect='auto')
    plt.subplot2grid((2, 3), (1, 1))
    plt.imshow(x_recon, interpolation='nearest', aspect='auto')
    plt.subplot2grid((2, 3), (1, 2))
    plt.imshow(middle_act, interpolation='nearest', aspect='auto')
    plt.subplot2grid((2, 3), (0, 2))
    #.imshow(mod_mid, interpolation='nearest', aspect='auto')
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    plt.draw()
    plt.show()
    # print dec_from_enc1
savemat('come_done_filtered.mat', {'collected_activity': collect_activity})

