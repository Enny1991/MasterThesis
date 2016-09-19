from __future__ import division
import time
import sys
#sys.path.append('/home/dneil/lasagne')
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, tanh, sigmoid
from lasagne. updates import adam
from scipy.io import loadmat, savemat
from tabulate import tabulate
#import time
import cPickle as pkl
import matlab.engine
if __name__ == "__main__":

    index_timit = int(sys.argv[1])
    tpe = int(sys.argv[2])  # 0 mse / 1 one_minus / 2 force_SIR
    p = float(sys.argv[3])
    NUM_UNITS_ENC = int(sys.argv[4])
    NUM_UNITS_DEC = int(sys.argv[5])
    third_layer = int(sys.argv[6])
    eng = matlab.engine.start_matlab()
    


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
        return batch_x, batch_m, batch_f

    fs = 16

    train_m = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWRa']
    train_f = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWRb']
    train_x = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWR']
    test_m = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWRatest']
    test_f = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWRbtest']
    test_x = loadmat('/Data/Dropbox/DATASETS/TSP/TSP_{}_SCR.mat'.format(index_timit))['PWRtest']

    #########
    #######
    n_features = train_m.shape[1]  # this time they are 512
    if tpe is 0 or 2:
        nonlin = sigmoid
    if tpe is 1:
        nonlin = sigmoid

    max_len = 50

    NUM_UNITS_ENC = 300
    NUM_UNITS_DEC = 300

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

    #l_concat = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)
    l_concat = lasagne.layers.ElemwiseSumLayer([l_dec_fwd, l_dec_bwd])
    l_drop_1 = lasagne.layers.DropoutLayer(l_concat, p=p)
    l_encoder_2_m = lasagne.layers.GRULayer(l_drop_1, num_units=NUM_UNITS_ENC)
    l_encoder_2_f = lasagne.layers.GRULayer(l_drop_1, num_units=NUM_UNITS_ENC)
    l_drop_2_m = lasagne.layers.DropoutLayer(l_encoder_2_m, p=p)
    l_drop_2_f = lasagne.layers.DropoutLayer(l_encoder_2_f, p=p)
    prev_m = l_drop_2_m
    prev_f = l_drop_2_f
    if third_layer is 1:
        l_encoder_3_m = lasagne.layers.GRULayer(l_drop_2_m, num_units=NUM_UNITS_ENC)
        l_encoder_3_f = lasagne.layers.GRULayer(l_drop_2_f, num_units=NUM_UNITS_ENC)
        l_drop_3_m = lasagne.layers.DropoutLayer(l_encoder_3_m, p=p)
        l_drop_3_f = lasagne.layers.DropoutLayer(l_encoder_3_f, p=p)
        prev_m = l_drop_2_m
        prev_f = l_drop_2_f
    l_decoder_m = lasagne.layers.GRULayer(prev_m, num_units=NUM_UNITS_DEC)
    l_decoder_f = lasagne.layers.GRULayer(prev_f, num_units=NUM_UNITS_DEC)
    
    #inter_out_m = lasagne.layers.get_output(l_decoder_m, inputs={l_in: x_sym}) 
    #inter_out_f = lasagne.layers.get_output(l_decoder_f, inputs={l_in: x_sym})
    l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
    #l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=500, nonlinearity=rectify)
    #l_drop_m = lasagne.layers.DropoutLayer(l_dense_m, p=p)
    l_dense_2_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=nonlin)
    l_out_m = lasagne.layers.ReshapeLayer(l_dense_2_m, (-1, max_len, n_features))

    l_reshape_f = lasagne.layers.ReshapeLayer(l_decoder_f, (-1, NUM_UNITS_DEC))
    #l_dense_f = lasagne.layers.DenseLayer(l_reshape_f, num_units=500, nonlinearity=rectify)
    #l_drop_f = lasagne.layers.DropoutLayer(l_dense_f, p=0.0)
    l_dense_2_f = lasagne.layers.DenseLayer(l_reshape_f, num_units=n_features, nonlinearity=nonlin)
    l_out_f = lasagne.layers.ReshapeLayer(l_dense_2_f, (-1, max_len, n_features))

    output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})
    output_f = lasagne.layers.get_output(l_out_f, inputs={l_in: x_sym})
    output_m_test = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym}, deterministic=True)
    output_f_test = lasagne.layers.get_output(l_out_f, inputs={l_in: x_sym}, deterministic=True)

    # here I divide the 3 different type of training
    if tpe is 0:
        loss_all_m = lasagne.objectives.squared_error(output_m * x_sym, m_sym) + \
                     lasagne.objectives.squared_error((1 - output_m) * x_sym, f_sym)
        loss_all_f = lasagne.objectives.squared_error(output_f * x_sym, f_sym) + \
                     lasagne.objectives.squared_error((1 - output_f) * x_sym, m_sym)
        loss_mean_m = T.mean(loss_all_m)
        loss_mean_f = T.mean(loss_all_f)
        loss_all_m_test = lasagne.objectives.squared_error(output_m_test * x_sym, m_sym) + \
                     lasagne.objectives.squared_error((1 - output_m_test) * x_sym, f_sym)
        loss_all_f_test = lasagne.objectives.squared_error(output_f_test * x_sym, f_sym) + \
                     lasagne.objectives.squared_error((1 - output_f_test) * x_sym, m_sym)
        loss_mean_m_test = T.mean(loss_all_m_test)
        loss_mean_f_test = T.mean(loss_all_f_test)

    if tpe is 1:
        loss_all_m = lasagne.objectives.squared_error(output_m * x_sym, m_sym) + \
                     lasagne.objectives.squared_error((1. - output_m) * x_sym, f_sym)
        loss_mean_m = T.mean(loss_all_m)
	loss_all_m_test = lasagne.objectives.squared_error(output_m_test * x_sym, m_sym) + \
                     lasagne.objectives.squared_error((1. - output_m_test) * x_sym, f_sym)
        loss_mean_m_test = T.mean(loss_all_m_test)

    if tpe is 2:
        loss_all_m = lasagne.objectives.squared_error(output_m * x_sym, m_sym) \
                     - 0.05 * lasagne.objectives.squared_error(output_m * x_sym, f_sym)
        loss_all_f = lasagne.objectives.squared_error(output_f * x_sym, f_sym) \
            - 0.05 * lasagne.objectives.squared_error(output_f * x_sym, m_sym)
        loss_mean_m = T.mean(loss_all_m)
        loss_mean_f = T.mean(loss_all_f)

    all_params_target_m = lasagne.layers.get_all_params([l_out_m])
    all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_mean_m, all_params_target_m)]
    all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 100)
    updates_target_m = adam(all_grads_target_m, all_params_target_m)

    train_model_m = theano.function([x_sym, m_sym, f_sym],
                                    [loss_mean_m, output_m],
                                    updates=updates_target_m,
                                    on_unused_input='ignore')

    test_model_m = theano.function([x_sym, m_sym, f_sym],
                                   [loss_mean_m_test, output_m],
                                   on_unused_input='ignore')

    if tpe is not 1:
        all_params_target_f = lasagne.layers.get_all_params([l_out_f])
        all_grads_target_f = [T.clip(g, -10, 10) for g in T.grad(loss_mean_f, all_params_target_f)]
        all_grads_target_f = lasagne.updates.total_norm_constraint(all_grads_target_f, 100)
        updates_target_f = adam(all_grads_target_f, all_params_target_f)
        train_model_f = theano.function([x_sym, f_sym, m_sym],
                                        [loss_mean_f, output_f],
                                        updates=updates_target_f,
                                        on_unused_input='ignore')

        test_model_f = theano.function([x_sym, f_sym, m_sym],
                                       [loss_mean_f_test, output_f],
                                       on_unused_input='ignore')

    num_min_batches = 100
    n_batch = 150
    epochs = 1
    for i in range(epochs):
        start_time = time.time()
        loss_train_m = 0
        loss_train_f = 0
        for j in range(10):
            batch_x, batch_m, batch_f = create_batches(n_batch)
            loss_m, _ = train_model_m(batch_x, batch_m, batch_f)
            loss_train_m += loss_m

            if tpe is not 1:
                loss_f, _ = train_model_f(batch_x, batch_f, batch_m)
                loss_train_f += loss_f
        #print 'M loss %.10f' % (loss_train_m / 10)
        #if tpe is not 1:
            #print 'F loss %.10f' % (loss_train_f / 10)

        batch_test_x, batch_test_m, batch_test_f = create_batches(100, False)
        loss_test_m, out_m = test_model_m(batch_test_x, batch_test_m, batch_test_f)
        if tpe is not 1:
            loss_test_f, out_f = test_model_f(batch_test_x, batch_test_f, batch_test_m)
        stop_time = time.time() - start_time
        #print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
        #print 'M loss TEST = %.10f ' % loss_test_m,
        #if tpe is not 1:
        #    print 'F loss TEST = %.10f ' % loss_test_f
        #print '-'*20

    # final test
    test_x, test_m, test_f = create_batches(100, False)
    l_m, out_m = test_model_m(test_x, test_m, test_f)
    out_out_m = out_m
    print 'TRAIN M = %.10f' % (loss_train_m / 10)
    print 'TEST M = %.10f' % l_m

    if tpe is not 1:
        l_f, out_f = test_model_f(test_x, test_f, test_m)
        out_out_f = out_f
        print 'TRAIN F = %.10f' % (loss_train_m / 10)
	print 'TEST F = %.10f' % l_f

    date = time.strftime("%H:%M_%d:%m:%Y")
    type_tr = ''
    pref = 'results/tsp/{}_{}_{}_{}_{}_{}_'.format(index_timit,tpe,p,NUM_UNITS_ENC,NUM_UNITS_DEC,third_layer)
    postf = ''
    if tpe is 0:
        savemat('results/tsp/{}_{}_{}_{}_{}_{}_special_TSP_{}.mat'.format(index_timit,tpe,p,NUM_UNITS_ENC,NUM_UNITS_DEC,third_layer), {'out_out_m': out_out_m, 'out_out_f': out_out_f})
        type_tr = 'special'
    if tpe is 1:
        savemat('results/tsp/{}_{}_{}_{}_{}_{}_one_minus_TSP_{}.mat'.format(index_timit,tpe,p,NUM_UNITS_ENC,NUM_UNITS_DEC,third_layer), {'out_out_m': out_out_m})
        type_tr = 'one_minus'
    if tpe is 2:
        savemat('results/tsp/{}_{}_{}_{}_{}_{}_force_SIR_TSP_{}.mat'.format(index_timit,tpe,p,NUM_UNITS_ENC,NUM_UNITS_DEC,third_layer), {'out_out_m': out_out_m, 'out_out_f': out_out_f})
        type_tr = 'force'
    # recon pipeline'results/tsp/3L_1000_one_minus_TSP_{}.mat'.format(index_timit)
    SNR, SIR, SDR, SAR, PESQ, STOI = eng.reconSPEC(index_timit, type_tr, 0, pref, '', nargout=6)
    SNR = np.array(SNR)
    PESQ = np.array(PESQ)
    SIR = np.array(SIR)
    SAR = np.array(SAR)
    SDR = np.array(SDR)
    STOI = np.array(STOI)
    
    table = [["SNR", SNR[0][0], SNR[1][0]],
             ["SIR", SIR[0][0], SIR[1][0]],
             ["SDR", SDR[0][0], SDR[1][0]],
             ["SAR", SAR[0][0], SAR[1][0]],
             ["PESQ", PESQ[0][0], PESQ[1][0]],
             ["STOI", STOI[0][0], STOI[1][0]]]
    headers = ["Measurements", "Male", "Female"]

    print tabulate(table, headers, tablefmt='grid')
