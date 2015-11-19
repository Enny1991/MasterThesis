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

n_epochs = 50
d = 784
batch_size = 500
layers = [200, 50, 10]
in_layers = []
out_layers = []
enc_layers = []

l_in = lasagne.layers.InputLayer(shape=(batch_size, d))
prev_dim = d

for layer in layers:
    in_layers += [l_in]
    enc = lasagne.layers.DenseLayer(l_in, num_units=layer, nonlinearity=rectify)
    drop = lasagne.layers.DropoutLayer(enc, p=0.5)
    enc_layers += [drop]
    dec = lasagne.layers.DenseLayer(drop, num_units=prev_dim, nonlinearity=identity, W=enc.W.T)
    out_layers += [dec]
    prev_dim = layer
    l_in = lasagne.layers.InputLayer(shape=(batch_size, layer))



def trainer_single(index, batch_size=500):

    x_sym = T.dmatrix()
    output = lasagne.layers.get_output(out_layers[index], inputs={in_layers[index]: x_sym})
    loss_eval = lasagne.objectives.squared_error(output, x_sym).sum()
    loss_eval /= (2.*batch_size)

    all_params = lasagne.layers.get_all_params(out_layers[index])
    updates = lasagne.updates.adam(loss_eval, all_params)

    return theano.function([x_sym], loss_eval, updates=updates)


x_sym = T.dmatrix()
total = train_x.shape[0]

for epoch in range(n_epochs):
    sub_train_x = train_x
    for layer in range(len(layers)):
        train_model = trainer_single(layer)
        for i in range(total/500):
            t_batch = sub_train_x[i*500:(i+1)*500]
            eval_train = train_model(t_batch)
        sub_train_x = lasagne.layers.get_output(enc_layers[layer], deterministic=True, inputs={in_layers[layer]: x_sym}).eval({x_sym: sub_train_x})
        print "Layer %i %.10f (time=%i)" % (layer, eval_train, epoch)

# connect everything
sub_test = test_x
for layer in range(len(layers)):
    sub_test = lasagne.layers.get_output(enc_layers[layer], deterministic=True, inputs={in_layers[layer]: x_sym}).eval({x_sym: sub_test})


out = sub_test
for layer in range(len(layers)):
    out = lasagne.layers.get_output(out_layers[len(layers) - layer - 1], inputs={enc_layers[len(layers) - layer - 1]: x_sym}).eval({x_sym: out})



dec = out
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
