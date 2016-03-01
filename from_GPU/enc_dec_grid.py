from __future__ import division
import sys
sys.path.append('/home/dneil/lasagne')
import theano.tensor as T
import theano
import lasagne
#from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from decoder_attention import LSTMAttentionDecodeFeedbackLayer
import os
import cPickle as pkl
import time
import pdb

# loading the data

train_x1 = loadmat('data/train_asr_grid_1.mat')['train_x1']
train_x2 = loadmat('data/train_asr_grid_2.mat')['train_x2']
train_x3 = loadmat('data/train_asr_grid_3.mat')['train_x3']
train_x = np.concatenate((train_x1, train_x2, train_x3))
train_y = loadmat('data/train_asr_grid_4.mat')['train_y']
test_x = loadmat('data/train_asr_grid_4.mat')['test_x']
test_y = loadmat('data/train_asr_grid_4.mat')['test_y']
test_x_mask = loadmat('data/train_asr_grid_5.mat')['test_x_mask']
train_x_mask = loadmat('data/train_asr_grid_5.mat')['train_x_mask']

# train_x = train_x[1::2]
# test_x = test_x[1::2]
# train_y = train_y[1::2]
# test_y = test_y[1::2]
# train_x_mask = train_x_mask[1::2]
# test_x_mask = test_x_mask[1::2]

print train_x.shape
print test_x.shape
print train_y.shape
print test_y.shape
print train_x_mask.shape
print test_x_mask.shape

# I'll put the dirctly padded and with sequences of labels in a matrix so it's easier
max_len = train_x.shape[1]
MFCCs = train_x.shape[2]  # should be 257
# we need to pad the sequences, no worries we can create masks so we can learn only on the 'real' inputs


# plot some examples
# plt.figure()
# plt.imshow(train_x[sample], interpolation='nearest', aspect='auto')
# plt.xlabel('MFCCs')
# plt.ylabel('Time')
# plt.title("{}".format(new_train_y[sample]))
# plt.show()


# Here I only train the model

# let's create some batches for the training
def get_batch(batch_size=3):
    idx = np.random.randint(0, len(train_x), size=batch_size)
    return train_x[idx], train_y[idx], train_x_mask[idx]

# let us create the model, in this case we will use a simple RNN encoder-decoder with attention
# this mean that our encoder feeds directly into the LSTM decoder that at the end of the inputs creates the output

BATCH_SIZE = 400
MAX_LENGHT_INPUT = max_len
MAX_LENGHT_OUPUT = 6

NUM_UNITS_ENC = 500
NUM_UNITS_DEC = 500
NUM_OUTS = 52  # see dict #
#print "MFCCs: {}".format(MFCCs)

#some data for debug
v_train_x, v_train_y, v_train_x_mask = get_batch()

x_sym = T.tensor3()
x_mask_sym = T.matrix()
y_sym = T.imatrix()

l_in = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT, MFCCs))
print "-"*15 + "Network" + "-"*16
print "Input Layer: {}".format(lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: v_train_x}).shape)

l_in_mask = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT))
print "Mask Input Layer: {}".format(lasagne.layers.get_output(l_in_mask, inputs={l_in_mask: x_mask_sym}).eval({x_mask_sym: v_train_x_mask}).shape)

l_enc = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_ENC, mask_input=l_in_mask)
print "GRU Encoder: {}".format(lasagne.layers.get_output(l_enc, inputs={l_in: x_sym, l_in_mask: x_mask_sym}).eval({x_sym: v_train_x, x_mask_sym: v_train_x_mask}).shape)

l_dec = LSTMAttentionDecodeFeedbackLayer(l_enc,
                                         num_units=NUM_UNITS_DEC,
                                         aln_num_units=52,
                                         n_decodesteps=MAX_LENGHT_OUPUT)

print "LSTM Decoder Layer: {}".format(lasagne.layers.get_output(l_dec, inputs={l_in: x_sym, l_in_mask: x_mask_sym}).eval({x_sym: v_train_x, x_mask_sym: v_train_x_mask}).shape)

# classic reshape voodoo
# we have an input that is batch_size, time_steps, num_units
# we want batch_size * time_steps, num_units
# so the new shape will be 2 (first, second)
# the second dimension should be the same as the third dimension of the original
# the first being -1 means that the dimension is infer to keep the total number unchanged
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))

print "Reshaped Layer: {}".format(lasagne.layers.get_output(l_reshape, inputs={l_in: x_sym, l_in_mask: x_mask_sym}).eval({x_sym: v_train_x, x_mask_sym: v_train_x_mask}).shape)

# dense layer now
l_softmax = lasagne.layers.DenseLayer(l_reshape,
                                      num_units=NUM_OUTS,
                                      nonlinearity=lasagne.nonlinearities.softmax)

print "Softmax Layer: {}".format(lasagne.layers.get_output(l_softmax, inputs={l_in: x_sym, l_in_mask: x_mask_sym}).eval({x_sym: v_train_x, x_mask_sym: v_train_x_mask}).shape)

# now the output should be (batch_size*steps ,num_outs)
# I'll reshape the output to match the teaching signal
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTS))

# now I need to calculate the loss
out_decoder = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_in_mask: x_mask_sym},
                                        deterministic=False)

loss = T.nnet.categorical_crossentropy(T.reshape(out_decoder, (-1, NUM_OUTS)),
                                       y_sym.flatten())

loss_mean = T.mean(loss)

argmax = T.argmax(out_decoder, axis=-1)
eq = T.eq(argmax, y_sym)
acc = eq.mean()

# train
all_params = lasagne.layers.get_all_params([l_out], trainable=True)

all_grads = [T.clip(g, -3, 3) for g in T.grad(loss_mean, all_params)]
all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)

updates = lasagne.updates.adam(all_grads, all_params)

train_func = theano.function([x_sym, y_sym, x_mask_sym],
                             [loss_mean, acc, out_decoder],
                             updates=updates,
                             allow_input_downcast=True)

test_func = theano.function([x_sym, y_sym, x_mask_sym],
                            [loss_mean, acc, l_dec.alpha, out_decoder],
                            allow_input_downcast=True)


# it's the moment to train
# gen some val data

val_x, val_y, val_x_mask = test_x, test_y, test_x_mask

samples_2_process = 600001
samples_processed = 0
val_acc = []
val_processed = []
batch_size_train = 400
loss_print = 1000
val_step = 8000
out_out = []
loss_along = []
print "-"*15 + "Training" + "-"*15

try:
    while samples_processed < samples_2_process:
        b_train_x, b_train_y, b_train_x_mask = get_batch(batch_size=batch_size_train)
        avg_loss, acc_train, _ = train_func(b_train_x, b_train_y, b_train_x_mask)
        loss_along += [avg_loss]
        samples_processed += batch_size_train
        if samples_processed % loss_print is 0:
            print "Loss batch = {}".format(avg_loss)
        if samples_processed % val_step is 0:
            loss_val, curr_acc, alphas, out = test_func(val_x, val_y, val_x_mask)
            val_acc += [curr_acc]
            val_processed += [samples_processed]
            out_out += [out]
            print "Validation Accuracy = {}".format(curr_acc)

except KeyboardInterrupt:
    pass

# let's print some stuff
# plt.figure()
# plt.plot(val_processed, val_acc)
# plt.ylabel('Validation Accuracy', fontsize=15)
# plt.xlabel('Processed samples', fontsize=15)
# plt.title('', fontsize=20)
# plt.grid('on')
# plt.show()
#
# plt.figure()
# for i in alphas[0]:
#     plt.plot(i)
# plt.show()
# for i in out_out[0]:
#     plt.imshow(i, interpolation='nearest', aspect='auto')
#     plt.show()

# save some cool stuff
date = time.strftime("%H:%M_%d:%m:%Y")
pkl.dump(lasagne.layers.get_all_param_values([l_out]),
                  open('models/asr_grid{}_{}'.format(date, NUM_UNITS_ENC), 'wb'))
savemat('out_LSTM_attention_{}'.format(date), {'out_out': out_out, 'val_acc': val_acc, 'loss_along': loss_along})

