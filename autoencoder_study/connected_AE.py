
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
        enc = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify, W=init.Uniform(0.001), b=None)
        Ws += [enc.W]
        drop = lasagne.layers.DropoutLayer(enc, p=0.0)
        prev_layer = drop
    idx = 1
    last_enc = prev_layer
    # I need to put here the mask
    mask = lasagne.layers.InputLayer(shape=(None, layers[-1]))
    mask_layer = lasagne.layers.ElemwiseMergeLayer([prev_layer, mask], merge_function=T.mul)
    prev_layer = mask_layer
    for layer in layers[-2::-1]:
        print layer
        dec = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify, W=Ws[-idx].T, b=None)
        idx += 1
        drop = lasagne.layers.DropoutLayer(dec, p=0.0)
        prev_layer = drop

    model = lasagne.layers.DenseLayer(prev_layer, num_units=s, nonlinearity=identity, W=Ws[0].T, b=None)

    x_sym = T.dmatrix()
    mask_sym = T.dmatrix()
    all_params = lasagne.layers.get_all_params(model)
    for i in all_params:
        print i
    output = lasagne.layers.get_output(model, inputs={l_in: x_sym, mask: mask_sym})
    loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()
    loss_eval /= (2.*batch_size)
    updates = lasagne.updates.adam(loss_eval, all_params)

    return l_in, model, last_enc, theano.function([x_sym, mask_sym], loss_eval, updates=updates), mask




####

rng = np.random.RandomState(42)
cor = loadmat('../data/complex_COR_5R.mat')['COR']
rec_cor_1 = np.zeros_like(cor)
rec_cor_2 = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq, d))
mask_a = np.ones((1, 2))
mask_b = np.ones((1, 2))
mask2 = np.ones((512, 2))
mask_a[:, 0] = 1.
mask_a[:, 1] = 0.
mask_b[:, 0] = 0.
mask_b[:, 1] = 1.
all_params = None
a2 = 2560
# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((64, d))
x_fake_2 = np.ones((32, d))
WW = np.zeros((time, 2560, 2))

n_epochs = 100
batch_size = 500
layers = [2]
Ws = []
collect_activity_a = np.zeros((time, rates / 2, freq))
collect_activity_b = np.zeros((time, rates / 2, freq))

l_in, model, last_enc, train_model, mask_l = createMLP(layers, a2)
# fig, axes = plt.subplots(nrows=2, ncols=3)

tot = 0

for t in range(time):
    x = cor[t, :, :]
    x_prime = np.zeros((freq * rates/2, 1))
    for rate in range(rates/2):
        d11 = x[:, rate]
        d1 = np.outer(d11, d11)
        d22 = x[:, rates/2 + rate]
        d2 = np.outer(d22, d22)
        x_prime[rate * freq:(rate+1) * freq, 0] = np.real((d11 + d22) / 2)
        c_mat[:, slice(rate * freq, (rate + 1) * freq)] = np.real((d1 + d2) / 2)
    x_prime = x_prime.transpose()
    # c_mat[-1, :] = x_prime
    # c_mat_lim = c_mat
    # c_mat_lim = (c_mat_lim - np.min(c_mat_lim)) / (np.max(c_mat_lim) - np.min(c_mat_lim))
    # c_mat_lim = c_mat_lim * 2 - 1

    for epoch in range(n_epochs):
        # c_mat_sc = c_mat[np.random.randint(0, freq+1, size=(freq + 1))]
        # c_mat_sc += rng.rand(c_mat_sc.shape[0], c_mat_sc.shape[1]) / (np.max(c_mat_sc) * 10)
        # for i in range(8):
        eval_train = train_model(c_mat, mask2)
    print " %.10f (time=%i)" % (eval_train, t)
    tot += eval_train
    # c_recon = lasagne.layers.get_output(model, {l_in: x_sym, mask_l: mask_sym}, deterministic=True).eval({x_sym: c_mat_lim, mask_sym: mask2})
    # middle_act = lasagne.layers.get_output(last_enc, {l_in: x_sym}, deterministic=True).eval({x_sym: x_prime})
    x_recon_a = np.reshape(lasagne.layers.get_output(model, {l_in: x_sym, mask_l: mask_sym}, deterministic=True).eval({x_sym: x_prime, mask_sym: mask_a})[-1], (5, 512))
    x_recon_b = np.reshape(lasagne.layers.get_output(model, {l_in: x_sym, mask_l: mask_sym}, deterministic=True).eval({x_sym: x_prime, mask_sym: mask_b})[-1], (5, 512))

    # high = np.zeros_like(middle_act)
    # mod_mid = copy.deepcopy(middle_act)
    # for row, i in zip(mod_mid, range(len(middle_act))):
    #     d = np.argsort(row)
    #     d_min = d[:len(d)/2]
    #     d_max = d[len(d)/2:]
    #     row[d_min] = 0
    #     row[d_max] = 1
    #     high[i, :] = row
    collect_activity_a[t, :, :] = x_recon_a
    collect_activity_b[t, :, :] = x_recon_b


    # plt.subplot2grid((2, 3), (0, 0))
    # plt.imshow(c_mat_lim, interpolation='nearest', aspect='auto')
    # plt.subplot2grid((2, 3), (0, 1))
    # plt.imshow(c_recon, interpolation='nearest', aspect='auto')
    # plt.subplot2grid((2, 3), (1, 0))
    # plt.imshow(x_prime, interpolation='nearest', aspect='auto')
    # plt.subplot2grid((2, 3), (1, 1))
    # plt.imshow(x_recon, interpolation='nearest', aspect='auto')
    # plt.subplot2grid((2, 3), (1, 2))
    # plt.imshow(middle_act, interpolation='nearest', aspect='auto')
    # plt.subplot2grid((2, 3), (0, 2))
    #.imshow(mod_mid, interpolation='nearest', aspect='auto')
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.draw()
    # plt.show()
    # print dec_from_enc1
    WW[t, :, :] = np.reshape(lasagne.layers.get_all_param_values(model)[0], (2560, 2))
savemat('complex_filtered_both.mat', {'W': WW, 'collected_activity_a': collect_activity_a, 'collected_activity_b': collect_activity_b})
print'average rec error %.10f' % tot
