from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
import numpy as np
import lasagne
import theano
import time
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
from scipy.io import loadmat, savemat
import time
import cPickle as pkl



def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def create_batches(n_samples, train=True):
    show = False
    if train:
        sel_mix = train_x
        sel_m = train_m
        sel_f = train_f
        #sel_mask_m = train_mask_m
        #sel_mask_f = train_mask_f
        batch_x = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_mask_m = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        batch_mask_f = np.zeros((n_samples, max_len, n_features)).astype(theano.config.floatX)
        idx = np.random.permutation(sel_mix.shape[0] - max_len)
        beg = idx[:n_samples]
        for i, b in enumerate(beg):
            batch_x[i] = sel_mix[b:b + max_len]
            batch_m[i, :, :] = sel_m[b:b + max_len]
            batch_f[i, :, :] = sel_f[b:b + max_len]
            #batch_mask_m[i, :, :] = sel_mask_m[b:b + max_len]
            #batch_mask_f[i, :, :] = sel_mask_f[b:b + max_len]

    else:
        sel_mix = test_x
        sel_m = test_m
        sel_f = test_f
        show = True
        batch_x = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        batch_m = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        batch_f = np.zeros((111, max_len, n_features)).astype(theano.config.floatX)
        for i in range(111):
            batch_x[i] = sel_mix[i*max_len:((i+1)*max_len)]
            batch_m[i, :, :] = sel_m[i*max_len:((i+1)*max_len)]
            batch_f[i, :, :] = sel_f[i*max_len:((i+1)*max_len)]
    return batch_x, batch_m, batch_f


data_m = loadmat('../../../data/m_f_book_SPEC.mat')['hunchback']
data_f = loadmat('../../../data/m_f_book_SPEC.mat')['screw']
#mask_m = loadmat('../../../data/m_f_book_SPEC.mat')['s_mask_m']
#mask_f = loadmat('../../../data/m_f_book_SPEC.mat')['s_mask_f']
data_mix = loadmat('../../../data/m_f_book_SPEC.mat')['hunchback_screw']

#########
# max_man = np.max(np.abs(data_m))
# max_fem = np.max(np.abs(data_f))
# data_mix /= np.max(np.abs(data_mix))
# data_m /= np.max(np.abs(data_m))
# data_f /= np.max(np.abs(data_f))
#######
n_features = data_m.shape[1]  # this time they are 512

print 'male shape {}'.format(data_m.shape)
print 'female shape {}'.format(data_f.shape)
print 'mix shape {}'.format(data_mix.shape)

max_len = 200
n_samples_train = 360000
n_samples_test = 22200
# # of samples is actually very high

# divide
train_x = data_mix[:n_samples_train]
train_m = data_m[:n_samples_train]
train_f = data_f[:n_samples_train]
#train_mask_m = data_m[:n_samples_train]
#train_mask_f = data_f[:n_samples_train]
test_x = data_mix[n_samples_train:]
test_m = data_m[n_samples_train:]
test_f = data_f[n_samples_train:]
#test_mask_m = data_m[n_samples_train:]
#test_mask_f = data_f[n_samples_train:]

# savemat('check.mat',{'x':test_x,'f':test_f,'m':test_m})

# create teaching signal

n_batch = 40
n_test = 60

ftrain_x, ftrain_m,ftrain_f = create_batches(10)
ftest_x, ftest_m, ftest_f = create_batches(10, False)
#
#
# print ftrain_x.shape
# print ftrain_m.shape
# print ftest_x.shape
# print ftest_m.shape

NUM_UNITS_ENC = 100
NUM_UNITS_DEC = 100

x_sym = T.tensor3()
mask_x_sym = T.matrix()
m_sym = T.tensor3()
f_sym = T.tensor3()
t_sym = T.tensor3()
mask_m_sym = T.tensor3()
mask_f_sym = T.tensor3()
n_sym = T.tensor3()
mask_n_sym = T.tensor3()


l_in = lasagne.layers.InputLayer(shape=(None, max_len, n_features))

l_dec_fwd = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_DEC, name='GRUDecoder', backwards=False)

l_encoder_2_m = lasagne.layers.GRULayer(l_dec_fwd, num_units=NUM_UNITS_ENC)
l_encoder_2_f = lasagne.layers.GRULayer(l_dec_fwd, num_units=NUM_UNITS_ENC)

l_decoder_m = lasagne.layers.GRULayer(l_encoder_2_m, num_units=NUM_UNITS_DEC)
l_decoder_f = lasagne.layers.GRULayer(l_encoder_2_f, num_units=NUM_UNITS_DEC)

l_reshape_m = lasagne.layers.ReshapeLayer(l_decoder_m, (-1, NUM_UNITS_DEC))
l_dense_m = lasagne.layers.DenseLayer(l_reshape_m, num_units=n_features, nonlinearity=rectify)
l_out_m = lasagne.layers.ReshapeLayer(l_dense_m, (-1, max_len, n_features))

l_reshape_f = lasagne.layers.ReshapeLayer(l_decoder_f, (-1, NUM_UNITS_DEC))
l_dense_f = lasagne.layers.DenseLayer(l_reshape_f, num_units=n_features, nonlinearity=rectify)
l_out_f = lasagne.layers.ReshapeLayer(l_dense_f, (-1, max_len, n_features))

# let's merge some shit all together
# l_sum = lasagne.layers.ElemwiseSumLayer([l_out_m, l_out_f])
#
# l_div_m = lasagne.layers.ElemwiseMergeLayer([l_out_m, l_sum], merge_function=T.true_div)
# l_div_f = lasagne.layers.ElemwiseMergeLayer([l_out_f, l_sum], merge_function=T.true_div)
#
# l_masking_m = lasagne.layers.ElemwiseMergeLayer([l_div_m, l_in], merge_function=T.mul)
# l_masking_f = lasagne.layers.ElemwiseMergeLayer([l_div_f, l_in], merge_function=T.mul)
#
# l_concat = lasagne.layers.ConcatLayer([l_masking_m, l_masking_f], axis=2)


output_m = lasagne.layers.get_output(l_out_m, inputs={l_in: x_sym})
output_f = lasagne.layers.get_output(l_out_f, inputs={l_in: x_sym})

# joint error function
# attempt to correct for nans
eps = 1e-8
masked_m = ((output_m + eps) / (output_m + output_f + eps)) * x_sym
# masked_m = T.switch(T.isnan(masked_m), 0., masked_m)
masked_f = ((output_f + eps) / (output_m + output_f + eps)) * x_sym
# masked_f = T.switch(T.isnan(masked_f), 0., masked_f)
# sub = T.switch(T.isnan(comp),0.,comp)

loss_all = T.mean(lasagne.objectives.squared_error(masked_m, m_sym)) + \
           T.mean(lasagne.objectives.squared_error(masked_f, f_sym))

#print loss_all.eval({x_sym: ftrain_x, m_sym: ftrain_m, f_sym: ftrain_f})
# - gamma * lasagne.objectives.squared_error(masked_f, m_sym)
# - gamma * lasagne.objectives.squared_error(masked_m, f_sym)

all_params_target_m = lasagne.layers.get_all_params([l_out_m])
all_grads_target_m = [T.clip(g, -10, 10) for g in T.grad(loss_all, all_params_target_m)]
all_grads_target_m = lasagne.updates.total_norm_constraint(all_grads_target_m, 10)

all_params_target_f = lasagne.layers.get_all_params([l_out_f])
all_grads_target_f = [T.clip(g, -10, 10) for g in T.grad(loss_all, all_params_target_f)]
all_grads_target_f = lasagne.updates.total_norm_constraint(all_grads_target_f, 10)


updates_target_m = adam(all_grads_target_m, all_params_target_m)
updates_target_f = adam(all_grads_target_f, all_params_target_f)


train_model_m = theano.function([x_sym, m_sym, f_sym],
                                loss_all,
                                updates=updates_target_m)

train_model_f = theano.function([x_sym, f_sym, m_sym],
                                loss_all,
                                updates=updates_target_f)

test_model_m = theano.function([x_sym, m_sym, f_sym],
                               [loss_all, output_m])
test_model_f = theano.function([x_sym, f_sym, m_sym],
                               [loss_all, output_f])

num_min_batches = 100
n_batch = 150
epochs = 100

for i in range(epochs):
    start_time = time.time()
    for j in range(10):
        batch_x, batch_m, batch_f = create_batches(n_batch)
        loss_train_m = train_model_m(batch_x, batch_m, batch_f)
        loss_train_f = train_model_f(batch_x, batch_f, batch_m)
        print 'joint (M) loss %.10f' % loss_train_m,
        print 'joint (F) loss %.10f' % loss_train_f
    batch_test_x, batch_test_m, batch_test_f = create_batches(100, False)
    loss_test_m, out_m = test_model_m(batch_test_x, batch_test_m, batch_test_f)
    loss_test_f, out_f = test_model_f(batch_test_x, batch_test_f, batch_test_m)
    stop_time = time.time() - start_time
    ma_m = ((out_m[0] + eps) / (out_m[0] + out_f[0] + eps)) * batch_test_x[0]
    ma_f = ((out_f[0] + eps) / (out_m[0] + out_f[0] + eps)) * batch_test_x[0]
    print ('-'*5 + ' epoch = %i ' + '-'*5 + ' time = %.4f ' + '-'*5) % (i, stop_time)
    print 'M loss TEST = %.10f ' % loss_test_m,
    print 'F loss TEST = %.10f ' % loss_test_f

# final test
test_x, test_m, test_f = create_batches(100, False)
l_m, out_m = test_model_m(test_x, test_m, test_f)
l_f, out_f = test_model_f(test_x, test_f, test_m)
out_out_m = out_m
out_out_f = out_f

print 'joint (M) TEST = %.10f' % l_m
print 'joint (F) TEST = %.10f' % l_f,
savemat('RR_2models_spec_dense_coupled.mat', {'out_out_m': out_out_m,
                                              'out_out_f': out_out_f,
                                              'max_m': np.max(np.abs(data_m)),
                                              'max_f': np.max(np.abs(data_f))})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_out_m]),
         open('RR_2models_spec_dense_coupled_m'.format(date, NUM_UNITS_ENC), 'wb'))
pkl.dump(lasagne.layers.get_all_param_values([l_out_f]),
         open('RR_2models_spec_dense_coupled_f'.format(date, NUM_UNITS_ENC), 'wb'))


