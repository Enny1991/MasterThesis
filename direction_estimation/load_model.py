import os
import pickle
import lasagne
import numpy as np
from scipy.io import loadmat
import theano.tensor as T
import theano
from matplotlib import pyplot as plt
from scipy.io import savemat

PARAM_EXTENSION = 'params'


np.random.seed(42)
data_x = loadmat('data/direction_dataset_noZERO.mat')['X_train_final_1']
data_y = loadmat('data/direction_dataset_noZERO.mat')['Y_train_1']
n_test = 128


def main(dd):

    # load hyperparameters
    h, eta, grad_clip, len_sample, n_dir = read_hyp('hyp_{}'.format(dd))
    # load model
    print("Load Network")
    load_l_in = lasagne.layers.InputLayer(shape=(None, len_sample, 2))

    # slice the las step to extract label
    load_l_forward_1 = lasagne.layers.LSTMLayer(
       load_l_in, h, grad_clipping=grad_clip,
       nonlinearity=lasagne.nonlinearities.tanh)

    #load_l_forward_2 = lasagne.layers.LSTMLayer(
    #    load_l_forward_1, h, grad_clipping=grad_clip,
    #    nonlinearity=lasagne.nonlinearities.tanh)

    load_l_forward_slice = lasagne.layers.SliceLayer(load_l_forward_1, -1, 1)

    load_l_out = lasagne.layers.DenseLayer(
        load_l_forward_slice, num_units=n_dir, W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax
    )
    read_model_data(load_l_out, 'model_{}'.format(dd))

    target_values = T.ivector('target_output')

    network_output = lasagne.layers.get_output(load_l_out)
    network_act = lasagne.layers.get_output(load_l_forward_1)
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values), dtype=theano.config.floatX)

    compute_cost = theano.function(
        [load_l_in.input_var, target_values], [cost, acc, network_output, network_act], allow_input_downcast=True)


    # test
    perm = np.random.permutation(len(data_x))
    perm_data_x = data_x[perm[:n_test]]
    perm_data_y = data_y[perm[:n_test]]
    y_test = np.zeros(n_test)
    x_test = np.zeros((n_test, len_sample, 2))
    for i in range(n_test):
        y_test[i] = perm_data_y[i] - 1  # mmmm...
        x_test[i] = perm_data_x[i][0:len_sample]

    # test
    cost_test, acc_test, output_test, network_act_test = compute_cost(x_test, y_test)
    savemat('outputs_raw.mat', {'output': output_test, 'input': x_test, 'labels': y_test})
    dump_results((output_test, y_test, x_test), dd)
    print("Final test cost = {}, acc = {}".format(cost_test, acc_test))


def dump_results(out, filename):
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'res')
    with open(filename, 'w') as f:
        pickle.dump(out, f)


def read_hyp(filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('models/', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('models/', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)

    savemat('dir_est_weights_raw.mat', {'data': data})

    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    date = '20:51_21:01:2016'
    main(date)
