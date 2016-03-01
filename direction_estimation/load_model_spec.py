import os
import pickle
import lasagne
import numpy as np
import scipy
from scipy.io import loadmat
import theano.tensor as T
import theano
from matplotlib import pyplot as plt
from scipy.io import savemat
PARAM_EXTENSION = 'params'


np.random.seed(42)
data_x = loadmat('data/direction_dataset_spec_stacked.mat')['XX']
data_y = loadmat('data/direction_dataset_spec_stacked.mat')['Y_train']
mask = loadmat('data/direction_dataset_spec_stacked.mat')['MASK']
n_test = 1000
n_batch = 128
nfft = 65


def main(dd):

    # load hyperparameters
    h, eta, grad_clip, len_sample, n_dir = read_hyp('hyp_{}'.format(dd))
    # load model
    print("Load Network")
    load_l_in = lasagne.layers.InputLayer(shape=(n_batch, len_sample, nfft * 4))
    l_mask = lasagne.layers.InputLayer(shape=(n_batch, len_sample))

    load_l_forward_1 = lasagne.layers.GRULayer(
       load_l_in, h, mask_input=l_mask, grad_clipping=grad_clip)

    # load_l_forward_2 = lasagne.layers.LSTMLayer(
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
    F = lasagne.layers.get_all_params(load_l_out)
    for f in F:
        print f.name,
        print f.shape
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    acc = T.mean(T.eq(T.argmax(network_output, axis=1), target_values), dtype=theano.config.floatX)

    compute_cost = theano.function(
        [load_l_in.input_var, target_values, l_mask.input_var], [cost, acc, network_output, network_act], allow_input_downcast=True)


    # test
    perm = np.random.permutation(len(data_x))
    perm_data_x = data_x[perm[:n_test]]
    perm_data_y = data_y[perm[:n_test]]
    perm_mask = mask[perm[:n_test]]
    y_test = np.zeros(n_test)
    x_test = np.zeros((n_test, len_sample, nfft * 4))
    mask_test = np.zeros((n_test, len_sample))
    for i in range(n_test):
        y_test[i] = perm_data_y[i]-1  # mmmm...
        x_test[i] = perm_data_x[i][0:len_sample]
        mask_test[i] = perm_mask[i]

    # test
    cost_test, acc_test, output_test, network_act_test = compute_cost(x_test, y_test, mask_test)
    savemat('outputs.mat', {'output': network_act_test, 'input': x_test, 'labels': y_test})
    # dump_results((output_test, y_test, x_test), dd)
    print("Final test cost = {}, acc = {}".format(cost_test, acc_test))

    # Let's look at the weights:
        # we shall have 2 sets of weights
        # GRU:
            # reset: Wxr - Whr
            # update: Wxu - Whu
            # hidden: Wxc - Whc
        # GRU (Slice) - dense

    W_hc = load_l_forward_1.W_hid_to_hidden_update.get_value()
    W_xc = load_l_forward_1.W_in_to_hidden_update.get_value()
    W_hr = load_l_forward_1.W_hid_to_resetgate.get_value()
    W_xr = load_l_forward_1.W_in_to_resetgate.get_value()
    W_hu = load_l_forward_1.W_hid_to_updategate.get_value()
    W_xu = load_l_forward_1.W_in_to_updategate.get_value()

    W_dense = load_l_out.W.get_value()


    # Show a weight matrix
    plt.figure(figsize=(14,7))
    plt.subplot(3, 2, 1)
    plt.imshow(W_hc, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_hc')
    #
    plt.subplot(3, 2, 2)
    plt.imshow(W_xc, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_xc')
    #
    plt.subplot(3, 2, 3)
    plt.imshow(W_hr, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_hr')
    #
    plt.subplot(3, 2, 4)
    plt.imshow(W_xr, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_xr')
    #
    plt.subplot(3, 2, 5)
    plt.imshow(W_hu, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_hu')
    #
    plt.subplot(3, 2, 6)
    plt.imshow(W_xu, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('W_xu')

    plt.show()

    scipy.io.savemat('weights.mat', mdict={'whr': W_hr,
                                           'wxr': W_xr,
                                           'whc': W_hc,
                                           'wxc': W_xc,
                                           'whu': W_hu,
                                           'wxu': W_xu})


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
    savemat('dir_est_weights.mat', {'data': data})
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    date = '09:49_30:10:2015'
    main(date)

