from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
import theano.tensor as T
import theano
import lasagne
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from lasagne.nonlinearities import rectify, tanh
from decoder_attention import LSTMAttentionDecodeFeedbackLayer
import os
import cPickle as pkl
import time
import pdb


#################### SOURCE SEPARATION ############################
# Here I just need to load the model, conversely later I could also optimize the two things together
index_timit = 2

def create_batches():
    sel_babf = test_bf
    sel_babm = test_bm
    sel_m = test_m
    sel_f = test_f
    Q = int(sel_babm.shape[0] / max_len)
    batch_babm = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
    batch_babf = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
    batch_m = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
    batch_f = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
    for i in range(Q):
        batch_babm[i] = sel_babm[i*max_len:((i+1)*max_len)]
        batch_babf[i] = sel_babf[i*max_len:((i+1)*max_len)]
        batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
        batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_babm, batch_babf, batch_m, batch_f

def load_model_index(x):
    return{
        1: ('force_SIR_GRID_m_14:23_09:01:2016_150_1',
            'force_SIR_GRID_f_14:23_09:01:2016_150_1'),
        2: ('force_SIR_GRID_m_14:47_09:01:2016_150_2',
            'force_SIR_GRID_f_14:47_09:01:2016_150_2'),
        3: ('force_SIR_GRID_m_15:10_09:01:2016_150_3',
            'force_SIR_GRID_f_15:10_09:01:2016_150_3'),
        4: ('force_SIR_GRID_m_15:35_09:01:2016_150_4',
            'force_SIR_GRID_f_15:35_09:01:2016_150_4'),
        5: ('force_SIR_GRID_m_15:57_09:01:2016_150_5',
            'force_SIR_GRID_f_15:57_09:01:2016_150_5'),
        6: ('force_SIR_GRID_m_16:18_09:01:2016_150_6',
            'force_SIR_GRID_f_16:18_09:01:2016_150_6'),
    }[x]

test_m = loadmat('../../../data/PCA/GRID corpus/12kHz/test_bab_grid_{}.mat'.format(index_timit))['PWRatest']
test_f = loadmat('../../../data/PCA/GRID corpus/12kHz/test_bab_grid_{}.mat'.format(index_timit))['PWRbtest']
test_bm = loadmat('../../../data/PCA/GRID corpus/12kHz/test_bab_grid_{}.mat'.format(index_timit))['PWRtestmale']
test_bf = loadmat('../../../data/PCA/GRID corpus/12kHz/test_bab_grid_{}.mat'.format(index_timit))['PWRtestfemale']


nonlin = rectify
n_features = 257
max_len = 50

NUM_UNITS_ENC = 150
NUM_UNITS_DEC = 150

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
l_encoder_2_f = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC)

l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_DEC)
l_decoder_f = lasagne.layers.GRULayer(l_encoder_2_f, num_units=NUM_UNITS_DEC)

l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=nonlin)
l_out_m = lasagne.layers.ReshapeLayer(l_dense_m, (-1, max_len, n_features))

l_reshape_f = lasagne.layers.ReshapeLayer(l_decoder_f, (-1, NUM_UNITS_DEC))
l_dense_f = lasagne.layers.DenseLayer(l_reshape_f, num_units=n_features, nonlinearity=nonlin)
l_out_f = lasagne.layers.ReshapeLayer(l_dense_f, (-1, max_len, n_features))

output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})
output_f = lasagne.layers.get_output(l_out_f, inputs={l_in: x_sym})


test_model_m = theano.function([x_sym],
                               [output_m],
                               on_unused_input='ignore')

test_model_f = theano.function([x_sym],
                               [output_f],
                               on_unused_input='ignore')


# Here I load
models = load_model_index(index_timit)
params_m = pkl.load(open(models[0], 'rb'))
params_f = pkl.load(open(models[1], 'rb'))
lasagne.layers.set_all_param_values(l_out_m, params_m)
lasagne.layers.set_all_param_values(l_out_f, params_f)


### PIPELINE
batch_test_babm, batch_test_babf, batch_test_m, batch_test_f = create_batches()
out_m = test_model_m(batch_test_babm)
out_f = test_model_f(batch_test_babf)

savemat('masks_bab_grid_2.mat', {'out_out_m':out_m[0],'out_out_f':out_f[0]})




