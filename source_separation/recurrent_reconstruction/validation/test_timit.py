from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
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

if __name__ == "__main__":
    index_timit = sys.argv[1]
    type = sys.argv[2]  # 0 mse / 1 one_minus / 2 force_SIR
    normalize = sys.argv[3]

    print index_timit
    print type
    print normalize