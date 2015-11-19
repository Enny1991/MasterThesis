import lasagne
import theano
import theano.tensor as T
import numpy as np
import time as tm
from lasagne.nonlinearities import rectify,identity, tanh
from scipy.io import loadmat, savemat
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import init
from lasagne.updates import adam
from matplotlib import pyplot as plt


cor = loadmat('../data/come_done_COR.mat')['COR']
rec_cor_1 = np.zeros_like(cor)
rec_cor_2 = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq+1, d))
mask = np.zeros((freq+1, 2))
mask2 = np.zeros((freq+1, 2))
mask[:, 0] = 1.
mask[:, 1] = 1.
all_params = None

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


l_in = lasagne.layers.InputLayer(shape=(batch_size, d))
prev_dim = d
prev_layer = l_in

for layer in layers:
    enc = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify)
    Ws += [enc.W]
    drop = lasagne.layers.DropoutLayer(enc, p=0.5)
    prev_layer = enc
idx = 1
for layer in layers[-2::-1]:
    print layer
    dec = lasagne.layers.DenseLayer(prev_layer, num_units=layer, nonlinearity=rectify, W=Ws[-idx].T)
    idx += 1
    prev_layer = dec
model = lasagne.layers.DenseLayer(prev_layer, num_units=d, nonlinearity=identity, W=Ws[0].T)


x_sym = T.dmatrix()
output = lasagne.layers.get_output(model, inputs={l_in: x_sym})
loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()
loss_eval /= (2.*batch_size)

all_params = lasagne.layers.get_all_params(model)
updates = lasagne.updates.adam(loss_eval, all_params)

train_model = theano.function([x_sym], loss_eval, updates=updates)

x_sym = T.dmatrix()

for t in range(time):
    x = cor[t, :, :]
    x_prime = np.zeros((d, 1))
    for rate in range(rates/2):
        d11 = x[:, rate]
        d1 = np.outer(d11, d11)
        d22 = x[:, rates/2 + rate]
        d2 = np.outer(d22, d22)
        x_prime[rate*freq:(rate+1)*freq, 0] = np.real((d11 + d22) / 2)
        c_mat[:-1, slice(rate * freq, (rate + 1) * freq)] = np.real((d1 + d2) / 2)
    x_prime = x_prime.transpose()
    c_mat[-1, :] = x_prime
    for epoch in range(n_epochs):
        # c_mat_sc = c_mat[np.random.randint(0, freq+1, size=(freq+1))]
        # for i in range(8):
        eval_train = train_model(c_mat)
        print "Layer %i %.10f (time=%i)" % (layer, eval_train, epoch)

    c_recon = lasagne.layers.get_output(model,{l_in: x_sym}, deterministic=True).eval({x_sym: c_mat})

    plt.figure
    plt.subplot(1, 2, 1)
    plt.imshow(c_mat, interpolation='nearest', aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(c_recon, interpolation='nearest', aspect='auto')
    plt.show()
    # print dec_from_enc1
savemat('come_done_filtered.mat', {'rec_cor_1': rec_cor_1, 'rec_cor_2': rec_cor_2})


#
# dec = lasagne.layers.get_output(model, inputs={l_in: x_sym}).eval({x_sym: test_x})
# fig = plt.figure()
# i = 0
# test_x_eval = test_x
# subset = np.random.randint(0, len(test_x_eval), size=50)
# img_out = np.zeros((28 * 2, 28 * len(subset)))
# x = np.array(test_x_eval)[np.array(subset)]
# for y in range(len(subset)):
#     x_a, x_b = 0 * 28, 1 * 28
#     x_recon_a, x_recon_b = 1 * 28, 2 * 28
#     ya, yb = y * 28, (y + 1) * 28
#     im = np.reshape(x[i], (28, 28))
#     im_recon = np.reshape(dec[subset[i]], (28, 28))
#     img_out[x_a:x_b, ya:yb] = im
#     img_out[x_recon_a:x_recon_b, ya:yb] = im_recon
#     i += 1
# m = plt.matshow(img_out, cmap='gray')
# plt.xticks(np.array([]))
# plt.yticks(np.array([]))
# plt.show()
