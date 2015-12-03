import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, identity
from lasagne.objectives import squared_error
from lasagne. updates import adam, adadelta, adagrad
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
import cPickle as pkl
import matlab.engine


class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        '''
        The input is expected to be a 2D tensor of shape
        (num_batch, num_features). The input is repeated
        n times such that the output will be
        (num_batch, n, num_features)
        '''
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        #repeat the input n times
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)


print 'Loading MATLAB...'
eng = matlab.engine.start_matlab()
eng.load_wavs_mocha(nargout=0)
theano.config.floatX = 'float64'

# create test
print 'Creating Test Set...'
[test_x, mask_test_x, target_test, test_n, mask_test] = eng.mixtures_test(nargout=5)
test_x = np.array(test_x)
mask_test_x = np.array(mask_test_x)
target_test = np.array(target_test)
test_n = np.array(test_n)
mask_test = np.array(mask_test)

max_len = test_x.shape[1]
n_features = test_x.shape[2]
n_batch = 12

NUM_UNITS_ENC = 50
NUM_UNITS_DEC = 50
NUM_UNITS_BOT = 15

x_sym = T.dtensor3()
mask_x_sym = T.dmatrix()
t_sym = T.dtensor3()
mask_t_sym = T.dtensor3()
n_sym = T.dtensor3()
mask_n_sym = T.dtensor3()

print 'Creating Model...'

l_in = lasagne.layers.InputLayer(shape=(n_batch, max_len, n_features))
l_mask = lasagne.layers.InputLayer(shape=(n_batch, max_len))

l_encoder = lasagne.layers.GRULayer(l_in,
                                    num_units=NUM_UNITS_ENC,
                                    mask_input=l_mask)
l_last_hid = lasagne.layers.SliceLayer(l_encoder, indices=-1, axis=1)

l_in_rep = RepeatLayer(l_last_hid, n=max_len)


# From here I need to create two decoders
# For target
l_decoder_target = lasagne.layers.GRULayer(l_in_rep, num_units=NUM_UNITS_DEC)
l_reshape_target = lasagne.layers.ReshapeLayer(l_decoder_target, (-1, NUM_UNITS_DEC))
l_out_target = lasagne.layers.DenseLayer(l_reshape_target, num_units=n_features, nonlinearity=rectify)
l_out_reshape_target = lasagne.layers.ReshapeLayer(l_out_target, (-1, x_sym.shape[1], n_features))
output_target = lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym})

# print lasagne.layers.get_output(l_out_reshape_target, inputs={l_in: x_sym, l_mask: mask_x_sym}).eval({x_sym:test_x,mask_x_sym:mask_test_x}).shape

loss_all_target = lasagne.objectives.squared_error(output_target * mask_t_sym, t_sym)

loss_mean_target = loss_all_target.mean()

# print loss_mean_target.eval({x_sym:test_x,mask_x_sym:mask_test_x, t_sym: target_train, mask_t_sym: mask_target_train})

all_params_target = lasagne.layers.get_all_params([l_out_reshape_target])
all_grads_target = [T.clip(g, -3, 3) for g in T.grad(loss_mean_target, all_params_target)]
all_grads_target = lasagne.updates.total_norm_constraint(all_grads_target, 3)
updates_target = adam(all_grads_target, all_params_target)

train_target = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                               loss_mean_target,
                               updates=updates_target)

test_target = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym],
                              [loss_mean_target, output_target])

# for noise
l_decoder_noise = lasagne.layers.GRULayer(l_in_rep, num_units=NUM_UNITS_DEC)
l_reshape_noise = lasagne.layers.ReshapeLayer(l_decoder_noise, (-1, NUM_UNITS_DEC))
l_out_noise = lasagne.layers.DenseLayer(l_reshape_noise, num_units=n_features, nonlinearity=rectify)
l_out_reshape_noise = lasagne.layers.ReshapeLayer(l_out_noise, (-1, x_sym.shape[1], n_features))
output_noise = lasagne.layers.get_output(l_out_reshape_noise, inputs={l_in: x_sym, l_mask: mask_x_sym})

loss_all_noise = lasagne.objectives.squared_error(output_noise * mask_n_sym, n_sym)
loss_mean_noise = loss_all_noise.mean()
all_params_noise = lasagne.layers.get_all_params([l_out_reshape_noise])
all_grads_noise = [T.clip(g, -3, 3) for g in T.grad(loss_mean_noise, all_params_noise)]
all_grads_noise = lasagne.updates.total_norm_constraint(all_grads_noise, 3)

updates_noise = adam(all_grads_noise, all_params_noise)

train_noise = theano.function([x_sym, mask_x_sym, n_sym, mask_n_sym],
                              loss_mean_noise,
                              updates=updates_noise)

test_noise = theano.function([x_sym, mask_x_sym, n_sym, mask_n_sym],
                             [loss_mean_noise, output_noise])

# train_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym, n_sym, mask_n_sym],
#                               [train_target, train_noise])
#
# test_model = theano.function([x_sym, mask_x_sym, t_sym, mask_t_sym, n_sym, mask_n_sym],
#                              [test_target, test_noise])

num_batches = 15
epochs = 50
print 'Training...'
for i in range(epochs):
    for j in range(num_batches):
        [train_x, mask_train_x, target_train, train_n, mask_train] = eng.random_mixtures_train(n_batch, nargout=5)
        train_x = np.array(train_x)
        mask_train_x = np.array(mask_train_x)
        target_train = np.array(target_train)
        train_n = np.array(train_n)
        mask_train = np.array(mask_train)
        loss_train_target = train_target(train_x,
                                         mask_train_x,
                                         target_train,
                                         mask_train)
        loss_train_noise = train_noise(train_x,
                                       mask_train_x,
                                       train_n,
                                       mask_train)
    loss_test_target, _ = test_target(test_x, mask_test_x, target_test, mask_test)
    loss_test_noise, _ = test_noise(test_x, mask_test_x, test_n, mask_test)
    print ('-'*5 + ' epoch = %i ' + '-'*5) % i
    print 'loss_train_target = %.6f \nloss_test_target = %.6f' % (loss_train_target, loss_test_target)
    print 'loss_train_noise = %.6f \nloss_test_noise = %.6f' % (loss_train_noise, loss_test_noise)

# final test
loss_test_target, output_target = test_target(test_x, mask_test_x, target_test, mask_test)
loss_test_noise, output_noise = test_noise(test_x, mask_test_x, test_n, mask_test)
savemat('filtered_samples_multiple.mat', {'output_target': output_target, 'output_noise': output_noise})
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_params(l_out_reshape_target),
         open('out_target_{}_{}_{}'.format(date, NUM_UNITS_ENC, NUM_UNITS_BOT), 'wb'))
pkl.dump(lasagne.layers.get_all_params(l_out_reshape_noise),
         open('out_noise_{}_{}_{}'.format(date, NUM_UNITS_ENC, NUM_UNITS_BOT), 'wb'))

eng.exit()


