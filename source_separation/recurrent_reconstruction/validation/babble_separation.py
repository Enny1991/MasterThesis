from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, tanh, sigmoid
from lasagne. updates import adam
from scipy.io import loadmat, savemat
import time
import cPickle as pkl

if __name__ == "__main__":

    index_timit = int(sys.argv[1])


    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]


    def create_batches(n_samples, train=True):
        show = False
        if train:
            sel_mix = train_x
            sel_m = train_m
            batch_x = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
            batch_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
            idx = np.random.permutation(sel_mix.shape[0] - max_len)
            beg = idx[:n_samples]
            for i, b in enumerate(beg):
                batch_x[i] = sel_mix[b:b + max_len]
                batch_m[i, :, :] = sel_m[b:b + max_len]

        else:
            sel_mix = test_x
            sel_m = test_m
            Q = int(sel_mix.shape[0] / max_len)
            batch_x = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
            batch_m = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
            for i in range(Q):
                batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
                batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
        return batch_x, batch_m

    train_m = loadmat('data/GRID_{}_BABBLE.mat'.format(index_timit))['PWRa']
    train_x = loadmat('data/GRID_{}_BABBLE.mat'.format(index_timit))['PWR']
    test_m = loadmat('data/GRID_{}_BABBLE.mat'.format(index_timit))['PWRatest']
    test_x = loadmat('data/GRID_{}_BABBLE.mat'.format(index_timit))['PWRtest']

    n_features = train_m.shape[1]  # this time they are 512
    nonlin = sigmoid

    max_len = 50

    NUM_UNITS_ENC = 500
    NUM_UNITS_DEC = 500

    x_sym = T.tensor3()
    mask_x_sym = T.matrix()
    m_sym = T.tensor3()
    f_sym = T.tensor3()
    mask_m_sym = T.tensor3()
    mask_f_sym = T.tensor3()
    n_sym = T.tensor3()
    mask_n_sym = T.tensor3()

    l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

    l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)
    l_dec_bwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=True)

    l_concat = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)

    l_encoder_2_m = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC)

    l_encoder_3_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_ENC)

    l_decoder_m = lasagne.layers.GRULayer(l_encoder_3_m, num_units=NUM_UNITS_DEC)

    l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
    l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=nonlin)
    l_out_m = lasagne.layers.ReshapeLayer(l_dense_m, (-1, max_len, n_features))

    output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})

    loss_all_m = lasagne.objectives.squared_error(output_m * x_sym, m_sym)
    loss_mean_m = T.mean(loss_all_m)

    all_params_target_m = lasagne.layers.get_all_params([l_out_m])
    all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean_m, all_params_target_m)]
    all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)
    updates_target_m = adam(all_grads_target_m, all_params_target_m)

    train_model_m = theano.function([x_sym, m_sym],
                                    [loss_mean_m, output_m],
                                    updates=updates_target_m,
                                    on_unused_input='ignore')

    test_model_m = theano.function([x_sym, m_sym],
                                   [loss_mean_m, output_m],
                                   on_unused_input='ignore')

    num_min_batches = 100
    n_batch = 120
    epochs = 50
    for i in range(epochs):
        start_time = time.time()
        loss_train_m = 0
        for j in range(10):
            batch_x, batch_m = create_batches(n_batch)
            loss_m, _ = train_model_m(batch_x, batch_m)
            loss_train_m += loss_m
        print 'M loss %.10f' % (loss_train_m / 10)

        batch_test_x, batch_test_m = create_batches(100, False)
        loss_test_m, out_m = test_model_m(batch_test_x, batch_test_m)
        stop_time = time.time() - start_time
        print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
        print 'M loss TEST = %.10f ' % loss_test_m
        print '-'*20

    # final test
    test_x, test_m = create_batches(100, False)
    l_m, out_m = test_model_m(test_x, test_m)
    out_out_m = out_m
    print 'M TEST = %.10f' % l_m

    savemat('results/babble/babble_mse_GRID_{}.mat'.format(index_timit), {'out_out_m': out_out_m})

