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
index_timit = 1



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

# test_m = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRatest']
# test_f = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRbtest']
# test_x = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}.mat'.format(index_timit))['PWRtest']
test_y = loadmat('../../../data/PCA/GRID corpus/12kHz/train_asr_grid_4.mat')['test_y']
test_x_mask = loadmat('../../../data/PCA/GRID corpus/12kHz/new_test_mask.mat')['test_x_mask']
full_test_x = loadmat('../../../data/PCA/GRID corpus/12kHz/full_test_x.mat')['full_test_x']
new_test_y = loadmat('../../../data/PCA/GRID corpus/12kHz/new_test_y.mat')['test_y']

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

#################### ASR ENC DEC ##################################

BATCH_SIZE = 100
MAX_LENGHT_INPUT = max_len
MAX_LENGHT_OUPUT = 6

NUM_UNITS_ENC = 400
NUM_UNITS_DEC = 400
NUM_OUTS = 52  # see dict #
MFCCs = 257
#print "MFCCs: {}".format(MFCCs)

x_sym = T.tensor3()
x_mask_sym = T.matrix()
y_sym = T.imatrix()

l_in = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT, MFCCs))

l_in_mask = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT))

l_enc = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_ENC, mask_input=l_in_mask)

l_dec = LSTMAttentionDecodeFeedbackLayer(l_enc,
                                         num_units=NUM_UNITS_DEC,
                                         aln_num_units=52,
                                         n_decodesteps=MAX_LENGHT_OUPUT)

l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))

l_softmax = lasagne.layers.DenseLayer(l_reshape,
                                      num_units=NUM_OUTS,
                                      nonlinearity=lasagne.nonlinearities.softmax)
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTS))
out_decoder = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_in_mask: x_mask_sym},
                                        deterministic=False)

loss = T.nnet.categorical_crossentropy(T.reshape(out_decoder, (-1, NUM_OUTS)),
                                       y_sym.flatten())

loss_mean = T.mean(loss)

argmax = T.argmax(out_decoder, axis=-1)
eq = T.eq(argmax, y_sym)
acc = eq.mean()

test_func = theano.function([x_sym, y_sym, x_mask_sym],
                            [loss_mean, acc, l_dec.alpha, out_decoder],
                            allow_input_downcast=True)

params = pkl.load(open('asr_grid10:46_11:01:2016_400', 'r'))
lasagne.layers.set_all_param_values(l_out, params)

### PIPELINE
sl = full_test_x[(index_timit-1)*60:index_timit*60]
mask_m = test_model_m(sl)
print mask_m[0].shape
out_out_m = sl * mask_m[0]
mask_f = test_model_f(full_test_x[(index_timit-1)*60:index_timit*60])
out_out_f = sl * mask_f[0]

print out_out_m.shape
# reshape for asr
asr_input = np.zeros((20, 266, 257))
for i in range(10):
    tmp = np.zeros((300, 257))
    tmp2 = np.zeros((300, 257))
    for j in range(6):
        tmp[j*50:(j+1)*50] = out_out_m[i*6 + j]
        tmp2[j*50:(j+1)*50] = out_out_f[i*6 + j]
        # tmp[j*50:(j+1)*50] = sl[i*6 + j]
        # tmp2[j*50:(j+1)*50] = sl[i*6 + j]
    asr_input[i * 2] = tmp[:266]
    asr_input[i * 2 + 1] = tmp2[:266]


# plt.figure()
# plt.imshow(asr_input[1],interpolation='nearest',aspect='auto')
# plt.show()
print asr_input.shape

loss, acc, _, ooo = test_func(asr_input,
                            new_test_y[(index_timit-1)*20:index_timit*20],
#                            np.ones((20, 266)))
                            test_x_mask[(index_timit-1)*20:index_timit*20])
savemat('ooo.mat', {'ooo': ooo})
print acc



