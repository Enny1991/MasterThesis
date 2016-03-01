from __future__ import print_function
import numpy as np
from scipy.io import loadmat
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import leaky_rectify, softmax
from theano import pp
import cPickle as pickle
import os
from matplotlib import pyplot as plt
import time

np.random.seed(42)
PARAM_EXTENSION = 'params'

# Rearrange the input from Matlab matricies
data_x = loadmat('data/direction_dataset_256_noPAD.mat')['X_train_final']
data_y = loadmat('data/direction_dataset_256_noPAD.mat')['Y_train']

n_dir = 9

h = 512
eta = 0.001
grad_clip = 100
epochs = 30
n_batch = 128
len_sample = 256
reg = 1e-3


# Extract test set
#
perm_test = np.random.permutation(len(data_x))
perm_data_x_test = data_x[perm_test[:n_batch]]
perm_data_y_test = data_y[perm_test[:n_batch]]
y_test = np.zeros(n_batch)
x_test = np.zeros((n_batch, len_sample, 2))

for i in range(n_batch):
    y_test[i] = perm_data_y_test[i]-1  # mmmm...
    x_test[i] = perm_data_x_test[i][0:len_sample]

# get rid of test data from the batch
data_x = data_x[perm_test[n_batch:]]
data_y = data_y[perm_test[n_batch:]]

epoch_size = np.floor(len(data_x) / n_batch)


def gen_input():
    perm = np.random.permutation(len(data_x))
    perm_data_x = data_x[perm[:n_batch]]
    perm_data_y = data_y[perm[:n_batch]]
    y = np.zeros(n_batch)
    x = np.zeros((n_batch, len_sample, 2))
    for i in range(n_batch):
        y[i] = perm_data_y[i]-1  # mmmm...
        x[i] = perm_data_x[i][0:len_sample]
    return x, np.array(y, dtype='int32')


def main(num_epochs=epochs):
    print("Building Network")
    l_in = lasagne.layers.InputLayer(shape=(num_epochs, len_sample, 2))
    # Could put here amask for the input layer:
    # l_mask = lasagne.layers.InputLayer(shape=(n_batch, len_sample))

    # slice the las step to extract label
    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, h, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, h, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

    l_out = DenseLayer(
        l_forward_slice, num_units=n_dir, W=lasagne.init.Normal(), nonlinearity=softmax
    )

    target_values = T.ivector('target_output')

    network_output = lasagne.layers.get_output(l_out)

    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values), dtype=theano.config.floatX)
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, eta)
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            [cost], updates=updates, allow_input_downcast=True)
    compute_cost = theano.function(
        [l_in.input_var, target_values], [cost, acc, network_output], allow_input_downcast=True)

    x_val, y_val = gen_input()

    def f(x):
        return {
            1:  "Epoch #{} [>_________]",
            2:  "Epoch #{} [=>________]",
            3:  "Epoch #{} [==>_______]",
            4:  "Epoch #{} [===>______]",
            5:  "Epoch #{} [====>_____]",
            6:  "Epoch #{} [=====>____]",
            7:  "Epoch #{} [======>___]",
            8:  "Epoch #{} [=======>__]",
            9:  "Epoch #{} [========>_]",
            10: "Epoch #{} [=========>]",
        }[x]

    print('Training')
    reprint = np.floor(epoch_size/10)
    cont = 0
    try:
        for epoch in range(num_epochs):
            print("Epoch #{} [__________]".format(epoch), end="\r")
            for e in range(epoch_size):
                if (e+1) % reprint == 0:
                    cont += 1
                    print(f(cont).format(epoch), end="\r")
                x, y = gen_input()
                train(x, y)
            cost_val, acc_val, _ = compute_cost(x_val, y_val)
            print("Epoch #{} [=========>] cost = {}, acc = {}".format(epoch, cost_val, acc_val))
            cont = 0
        cost_test, acc_test, output_test = compute_cost(x_test, y_test)
        date = time.strftime("%H:%M_%d:%m:%Y")
        write_model_data(l_out, 'model_{}'.format(date))
        list_hyp = (
            h,
            eta,
            grad_clip,
            len_sample,
            n_dir
        )
        dump_param(list_hyp,'hyp_{}'.format(date))
        print("Final test cost = {}, acc = {}".format(cost_test, acc_test))

        # Now I'd like to plot some output to see what happens with the cross entropy
        # the output_test should be N-by-F
        # selected_out = 3
        # print(output_test[selected_out])
        # # Plot the result
        # width = 0.35
        # plt.figure(figsize=(14, 7))
        # plt.bar(np.arange(0-width, 8+width), output_test[selected_out], width)
        # plt.xticks(np.arange(0-width,8)+width/2., [str(i) for i in range(0, 9)])        #plt.xlim([0,1])
        # plt.grid(which='both')
        # plt.show()
    except KeyboardInterrupt:
        pass


def dump_param(list_hyp, filename):
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(list_hyp, f)


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
