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

    index_timit = sys.argv[1]
    tpe = int(sys.argv[2])  # 0 mse / 1 one_minus / 2 force_SIR
    normalize = int(sys.argv[3])


    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]


    def create_batches(n_samples, train=True):
        show = False
        if train:
            sel_mix = train_x
            sel_m = train_m
            sel_f = train_f
            batch_x = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
            batch_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
            batch_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
            idx = np.random.permutation(sel_mix.shape[0] - max_len)
            beg = idx[:n_samples]
            for i, b in enumerate(beg):
                batch_x[i] = sel_mix[b:b + max_len]
                batch_m[i, :, :] = sel_m[b:b + max_len]
                batch_f[i, :, :] = sel_f[b:b + max_len]

        else:
            sel_mix = test_x
            sel_m = test_m
            sel_f = test_f
            Q = int(sel_mix.shape[0] / max_len)
            batch_x = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
            batch_m = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
            batch_f = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
            for i in range(Q):
                batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
                batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
                batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
        ibm_mask = batch_m > batch_f
        return batch_x, batch_m, batch_f, ibm_mask.astype('int32')


    train_m = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRa']
    train_f = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRb']
    train_x = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWR']
    test_m = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRatest']
    test_f = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRbtest']
    test_x = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRtest']

    #########
    if normalize is 1:
        mmm = max(np.max(np.abs(train_m)),
                  np.max(np.abs(train_f)),
                  np.max(np.abs(train_x)),
                  np.max(np.abs(test_m)),
                  np.max(np.abs(test_f)),
                  np.max(np.abs(test_x)))
        train_m /= mmm
        train_f /= mmm
        train_x /= mmm
        test_m /= mmm
        test_f /= mmm
        test_x /= mmm
    #######
    n_features = train_m.shape[1]  # this time they are 512
    if tpe is 0 or 2:
        nonlin = rectify
    if tpe is 1:
        nonlin = tanh

    max_len = 50

    debug_batch_x, debug_batch_m, debug_batch_f, debug_batch_mask = create_batches(10)

    NUM_UNITS_ENC = 150
    NUM_UNITS_DEC = 150

    x_sym = T.tensor3()
    mask_x_sym = T.matrix()
    y_sym = T.itensor3()
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

    l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_DEC)

    l_reshape = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, [2]))

    print "Reshape layer: {}".format(lasagne.layers.get_output(l_reshape, inputs={l_in: x_sym}).eval({x_sym: debug_batch_x}).shape)


    l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=257, nonlinearity=sigmoid)

    print "dense layer : {}".format(lasagne.layers.get_output(l_dense, inputs={l_in: x_sym}).eval({x_sym: debug_batch_x}).shape)


    l_out = lasagne.layers.ReshapeLayer(l_dense, (x_sym.shape[0], -1, 257))

    print "OUT: {}".format(lasagne.layers.get_output(l_out, inputs={l_in: x_sym}).eval({x_sym: debug_batch_x}).shape)

    # now I need to calculate the loss
    out_decoder = lasagne.layers.get_output(l_out, inputs={l_in: x_sym},
                                        deterministic=False)

    loss_mean = T.mean(lasagne.objectives.squared_error(out_decoder, y_sym))



    all_params_target_m = lasagne.layers.get_all_params([l_out])
    all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean, all_params_target_m)]
    all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)
    updates_target_m = adam(all_grads_target_m, all_params_target_m)

    train_model_m = theano.function([x_sym, y_sym],
                                    [loss_mean, out_decoder],
                                    updates=updates_target_m,
                                    on_unused_input='ignore')

    test_model_m = theano.function([x_sym, y_sym],
                                   [loss_mean, out_decoder],
                                   on_unused_input='ignore')

    num_min_batches = 100
    n_batch = 100
    epochs = 75
    for i in range(epochs):
        start_time = time.time()
        loss_train_m = 0
        loss_train_f = 0
        for j in range(10):
            batch_x, batch_m, batch_f, batch_mask = create_batches(n_batch)
            loss_m, _ = train_model_m(batch_x, batch_mask)
            loss_train_m += loss_m

        print 'M classification %.10f' % loss_m

        batch_test_x, batch_test_m, batch_test_f, batch_test_mask = create_batches(100, False)
        loss_test_m, out_m = test_model_m(batch_test_x, batch_test_mask)

        stop_time = time.time() - start_time
        print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
        print 'M loss TEST = %.10f ' % loss_test_m,
        print '-'*20

    # final test
    test_x, test_m, test_f, test_mask = create_batches(100, False)
    l_m, out_m = test_model_m(test_x, test_mask)
    out_out_m = out_m
    print 'M TEST = %.10f' % l_m

    date = time.strftime("%H:%M_%d:%m:%Y")

    savemat('results/mse_GRID_{}.mat'.format(index_timit), {'out_out_m': out_out_m})

