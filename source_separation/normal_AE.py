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

    return (x_train, targets_train), (x_test, targets_test), (x_valid, targets_valid)


(train_x, train_t), (test_x, test_t), (valid_x, valid_t) = load_mnist()
total = train_x.shape[0]
# from here we apply our data
d = 784
first_latent = d * 2
latent = 10
mask = np.ones((500, latent))
mask2 = np.ones((10000, latent))
# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((500, d))
# create the model UGLY
l_in_1 = lasagne.layers.InputLayer(shape=(500, d))
first_enc = lasagne.layers.DenseLayer(l_in_1, num_units=first_latent, nonlinearity=rectify)
sec_enc = lasagne.layers.DenseLayer(first_enc, num_units=latent, nonlinearity=rectify)

mask_input = lasagne.layers.InputLayer(shape=(500, latent))
merged_layer = lasagne.layers.ElemwiseMergeLayer([sec_enc, mask_input], merge_function=T.mul)

first_dec = lasagne.layers.DenseLayer(merged_layer, num_units=first_latent, nonlinearity=rectify, W=sec_enc.W.T)
out = lasagne.layers.DenseLayer(first_dec, num_units=d, nonlinearity=identity, W=first_enc.W.T)

x = T.dmatrix()
all_params = lasagne.layers.get_all_params(out)

output = lasagne.layers.get_output(out, inputs={l_in_1: x_sym, mask_input: mask_sym})
loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()

loss_eval /= (2.*500)


updates = lasagne.updates.adam(loss_eval, all_params)

train_model = theano.function([x_sym, mask_sym], loss_eval, updates=updates)

total = train_x.shape[0]

for epoch in range(50):
    for i in range(total/500):
        t_batch = train_x[i*500:(i+1)*500]
        eval_train = train_model(t_batch, mask)
    print "Layer 1 %.10f (time=%i)" % (eval_train, epoch)

dec = lasagne.layers.get_output(out, inputs={l_in_1: x_sym, mask_input: mask_sym}).eval({x_sym: test_x, mask_sym: mask2})

fig = plt.figure()
i = 0
test_x_eval = test_x
subset = np.random.randint(0, len(test_x_eval), size=50)
img_out = np.zeros((28 * 2, 28 * len(subset)))
x = np.array(test_x_eval)[np.array(subset)]
for y in range(len(subset)):
    x_a, x_b = 0 * 28, 1 * 28
    x_recon_a, x_recon_b = 1 * 28, 2 * 28
    ya, yb = y * 28, (y + 1) * 28
    im = np.reshape(x[i], (28, 28))
    im_recon = np.reshape(dec[subset[i]], (28, 28))
    img_out[x_a:x_b, ya:yb] = im
    img_out[x_recon_a:x_recon_b, ya:yb] = im_recon
    i += 1
m = plt.matshow(img_out, cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()
