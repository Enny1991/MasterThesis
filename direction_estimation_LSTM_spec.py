from __future__ import print_function
import numpy as np
from scipy.io import loadmat
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import leaky_rectify, softmax
import cPickle as pickle
import os
import time

np.random.seed(42)
PARAM_EXTENSION = 'params'

# Rearrange the input from Matlab matricies
data_x = loadmat('data/direction_dataset_spec_stacked.mat')['XX']
data_y = loadmat('data/direction_dataset_spec_stacked.mat')['Y_train']
mask = loadmat('data/direction_dataset_spec_stacked.mat')['MASK']

n_dir = 9

h = 512
eta = 0.01
grad_clip = 100
epochs = 30
n_batch = 128
len_sample = 207
reg = 1e-3
nfft = 65


# I implement 10-fold CV for eta and h
# I need to first extract the test set and only THEN separate in 10 buckets

# Extract test set
n_test = 200  # 10% for test

perm_test = np.random.permutation(len(data_x))
data_x = data_x[perm_test]
data_y = data_y[perm_test]
mask = mask[perm_test]
perm_data_x_test = data_x[:n_test]
perm_data_y_test = data_y[:n_test]
perm_data_mask_test = mask[:n_test]
y_test = np.zeros(n_test)
x_test = np.zeros((n_test, len_sample, nfft * 4))
mask_test = np.zeros((n_test, len_sample))
perm_data_x_val = data_x[n_test:2*n_test]
perm_data_y_val = data_y[n_test:2*n_test]
perm_data_mask_val = mask[n_test:2*n_test]
y_val = np.zeros(n_test)
x_val = np.zeros((n_test, len_sample, nfft * 4))
mask_val = np.zeros((n_test, len_sample))
data_x = data_x[2*n_test:]
data_y = data_y[2*n_test:]
mask = mask[2*n_test:]
y_train = np.zeros(len(data_y))
x_train = np.zeros((len(data_x), len_sample, nfft * 4))
mask_train = np.zeros((len(data_x), len_sample))


for i in range(n_test):
    y_test[i] = perm_data_y_test[i]-1  # mmmm...
    x_test[i] = perm_data_x_test[i]
    mask_test[i] = perm_data_mask_test[i]
    y_val[i] = perm_data_y_val[i]-1  # mmmm...
    x_val[i] = perm_data_x_val[i]
    mask_val[i] = perm_data_mask_val[i]


for i in range(len(data_x)):
    y_train[i] = data_y[i]-1  # mmmm...
    x_train[i] = data_x[i]
    mask_train[i] = mask[i]


epoch_size = (np.floor(len(data_x) / n_batch)).astype(np.int64)


def gen_input(CV=False, k=0, return_validation=False):
    if CV:
        # need to extract
        samples_per_bucket = np.floor(data_x/k)
        reduced_x = np.concatenate((data_x[0:k*samples_per_bucket], data_x[(k+1)*samples_per_bucket:]), axis=0)
        reduced_y = np.concatenate((data_y[0:k*samples_per_bucket], data_y[(k+1)*samples_per_bucket:]), axis=0)
        validation_x = data_x[k*samples_per_bucket:(k+1)*samples_per_bucket]
        validation_y = data_y[k*samples_per_bucket:(k+1)*samples_per_bucket]
        perm = np.random.permutation(len(data_x))
        perm_data_x = reduced_x[perm[:n_batch]]
        perm_data_y = reduced_y[perm[:n_batch]]
        y = np.zeros(n_batch)
        x = np.zeros((n_batch, len_sample, 2))
        if return_validation:
            for i in range(n_batch):
                y[i] = perm_data_y[i]-1  # mmmm...
                x[i] = perm_data_x[i][0:len_sample]
            return x, np.array(y, dtype='int32')
        else:
            for i in range(len(validation_x)):
                y[i] = validation_y[i]-1  # mmmm...
                x[i] = validation_x[i][0:len_sample]
            return x, np.array(y, dtype='int32')
    else:
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
    l_in = lasagne.layers.InputLayer(shape=(n_batch, len_sample, nfft * 4))  # 4 = 2*RE + 2*IM / None is for variable length of the input
    # Could put here a mask for the input layer:
    l_mask = lasagne.layers.InputLayer(shape=(n_batch, len_sample))

    # slice the las step to extract label
    l_forward_1 = lasagne.layers.GRULayer(
        l_in, h, mask_input=l_mask, grad_clipping=grad_clip)

    #l_forward_2 = lasagne.layers.LSTMLayer(
    #    l_forward_1, h, grad_clipping=grad_clip,
    #    nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)

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
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            [cost], updates=updates, allow_input_downcast=True)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], [cost, acc, network_output], allow_input_downcast=True)

    # x_val, y_val = gen_input()

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
                # x, y = gen_input()
                train(x_train[e*n_batch:(e+1)*n_batch], y_train[e*n_batch:(e+1)*n_batch], mask_train[e*n_batch:(e+1)*n_batch])
            cost_val, acc_val, _ = compute_cost(x_val, y_val, mask_val)
            print("Epoch #{} [=========>] cost = {}, acc = {}".format(epoch, cost_val, acc_val))
            cont = 0
        cost_test, acc_test, output_test = compute_cost(x_test, y_test, mask_test)
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

