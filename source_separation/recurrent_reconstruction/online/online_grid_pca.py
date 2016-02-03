from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
sys.path.append('/Users/enea/PycharmProjects/Thesis/source_separation/recurrent_reconstruction/online/')
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, tanh
from lasagne. updates import adam
from scipy.io import loadmat, savemat
import time
import cPickle as pkl
from pyspectre import *
import pyaudio
import wave
from matplotlib import pyplot as plt


def project(high_dim, v):
    v50 = v[:, :24]
    return np.dot(high_dim, v50).T

def reproject(low_dim, v):
    filler = np.zeros((low_dim.shape[0], 233))
    tomult = np.concatenate((low_dim, filler), axis=1)
    return np.dot(tomult, v.T).T

def wiener_filter(mix, a, b):
    ma = a**2 / (a**2 + b**2) * mix
    mb = b**2 / (a**2 + b**2) * mix
    ma[np.isnan(ma)] = 0.
    mb[np.isnan(mb)] = 0.
    return ma, mb

def create_batches(mix, a, b):
        sel_mix = mix
        sel_m = a
        sel_f = b
        Q = int(sel_mix.shape[0] / max_len)
        batch_x = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((Q, max_len, n_features)).astype(theano.config.floatX)
        for i in range(Q):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
            batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
        return batch_x, batch_m, batch_f

if __name__ == "__main__":

    #LOAD STUFF TO RECONSTRUCT
    GRID_1_locale = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_1_locale.mat')
    GRID_1_pca_locale = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_1_pca_locale.mat')
    GRID_1_pca = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_1_pca.mat')

    train_m = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWRa']
    train_f = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWRb']
    train_x = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWR']
    test_m = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWRatest']
    test_f = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWRbtest']
    test_x = loadmat('../../../data/PCA/GRID corpus/12kHz/GRID_{}_pca.mat'.format(1))['lPWRtest']


    mix = GRID_1_pca['lPWRtest']
    a = GRID_1_pca['lPWRatest']
    b = GRID_1_pca['lPWRbtest']
    mp = GRID_1_locale['mp']
    v = GRID_1_pca_locale['V']
    original = GRID_1_locale['PWRtest']
    test1 = GRID_1_locale['test1']
    test2 = GRID_1_locale['test2']
    test1 = test1[:, 0]
    test2 = test2[:, 0]

    mmm = max(np.max(np.abs(train_m)),
              np.max(np.abs(train_f)),
              np.max(np.abs(train_x)),
              np.max(np.abs(test_m)),
              np.max(np.abs(test_f)),
              np.max(np.abs(test_x)))
    # mix /= mmm
    max_len = 50
    Q = int(original.shape[0] / max_len)
    mp_batch = np.zeros((Q, 512, max_len))
    or_batch = np.zeros((Q, 257, max_len))
    for i in range(Q):
        mp_batch[i] = mp[:, i * max_len: (i + 1) * max_len]
        or_batch[i] = original[i * max_len: (i + 1) * max_len, :].T

    # wanna create batches

    n_features = 24
    mix_batch, a_batch, b_batch = create_batches(mix, a, b)
    #########

    nonlin = rectify

    NUM_UNITS_ENC = 50
    NUM_UNITS_DEC = 50

    init_fwd = T.matrix()
    init_bwd = T.matrix()
    init_enc_m = T.matrix()
    init_enc_f = T.matrix()
    init_dec_m = T.matrix()
    init_dec_f = T.matrix()

    x_sym = T.tensor3()
    mask_x_sym = T.matrix()
    m_sym = T.tensor3()
    f_sym = T.tensor3()
    mask_m_sym = T.tensor3()
    mask_f_sym = T.tensor3()
    n_sym = T.tensor3()
    mask_n_sym = T.tensor3()

    l_in = lasagne.layers.InputLayer(shape=(1, max_len, n_features))

    l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC,
                                        name='GRUDecoder',
                                        backwards=False, learn_init=True)
    l_dec_bwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC,
                                        name='GRUDecoder',
                                        backwards=True, learn_init=True)

    l_concat = lasagne.layers.ConcatLayer([l_dec_fwd, l_dec_bwd], axis=2)

    l_encoder_m = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC, learn_init=True)
    l_encoder_f = lasagne.layers.GRULayer(l_concat, num_units=NUM_UNITS_ENC, learn_init=True)

    l_decoder_m = lasagne.layers.GRULayer(l_encoder_m, num_units=n_features, learn_init=True)
    l_decoder_f = lasagne.layers.GRULayer(l_encoder_f, num_units=n_features, learn_init=True)

    # save last act
    output_m = lasagne.layers.get_output(l_decoder_m, inputs={l_in: x_sym})
    output_f = lasagne.layers.get_output(l_decoder_f, inputs={l_in: x_sym})

    # out_enc_m = lasagne.layers.get_output(l_encoder_m, inputs={l_in: x_sym})
    # out_enc_f = lasagne.layers.get_output(l_encoder_f, inputs={l_in: x_sym})
    #
    # out_fwd = lasagne.layers.get_output(l_dec_fwd, inputs={l_in: x_sym})
    # out_bwd = lasagne.layers.get_output(l_dec_bwd, inputs={l_in: x_sym})

    test_model_m = theano.function([x_sym],
                                   [output_m, output_f], on_unused_input='warn')

    params_m = pkl.load(open('real_mse_grid_m_11:15_19:12:2015_50_1_pca_direct', 'r'))
    params_f = pkl.load(open('real_mse_grid_f_11:15_19:12:2015_50_1_pca_direct', 'r'))
    lasagne.layers.set_all_param_values(l_decoder_m, params_m)
    lasagne.layers.set_all_param_values(l_decoder_f, params_f)
    out_low_dim_a = np.zeros((Q * max_len, n_features))
    out_low_dim_b = np.zeros((Q * max_len, n_features))

    #### RUNNING MODEL
    # To get it to run online it could be an idea to set the internal activity as the previous one,
    # since it has been trained on random sequences is should be possible to have this working
    # The levels to save are:
    # - GRU fwd
    # - GRU bwd
    # - Encoder M
    # - Encoder F
    # - Decoder M
    # - Decoder F

    # invert spec to raw
    init_0 = np.zeros((1, max_len, NUM_UNITS_DEC))
    in_fwd = init_0
    in_bwd = init_0
    in_enc_m = init_0
    in_enc_f = init_0
    out_m = init_0
    out_f = init_0
    for i in range(mix_batch.shape[0]):
        # for j in range(max_len / 2):
        out_m, out_f = test_model_m(mix_batch[i].reshape((1, max_len, n_features)) / mmm)
        j = 0
        out_low_dim_a[i * max_len + j:(i + 1) * max_len + j] = out_m[0]
        out_low_dim_b[i * max_len + j:(i + 1) * max_len + j] = out_f[0]

    print out_low_dim_a.shape
    high_m = reproject(out_low_dim_a, v)
    high_f = reproject(out_low_dim_b, v)
    print high_m.shape
    # plt.figure()
    # plt.imshow(high_m)
    # plt.show()
    filt_m, filt_f = wiener_filter(original.T, high_m, high_f)
    # plt.figure()
    # plt.imshow(filt_m)
    # plt.show()
    recon_m = inv_spectre(filt_m, mp)
    recon_f = inv_spectre(filt_f, mp)
    print snr_audio(test1, recon_m)
    print snr_audio(test2, recon_f)
    savemat('out.mat', {'recon_m': recon_m, 'recon_f':  recon_f})

    # bytestream = recon_m.tobytes()
    # pya = pyaudio.PyAudio()
    # stream = pya.open(format=pya.get_format_from_width(width=2), channels=1, rate=12000, output=True)
    # stream.write(bytestream)
    # stream.stop_stream()
    # stream.close()
    # pya.terminate()
        # write to output
        #






