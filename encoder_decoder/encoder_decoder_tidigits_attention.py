import theano.tensor as T
import theano
import lasagne
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from decoder_attention import LSTMAttentionDecodeFeedbackLayer
import os
import cPickle as pickle
import time
import pdb

theano.config.floatX = 'float64'
PARAM_EXTENSION = 'pkl'
np.random.seed(42)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('.'
                            '', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def look_up(x):
    return{
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
        '6': 5,
        '7': 6,
        '8': 7,
        '9': 8,
        'O': 9,
        'Z': 10,
        '#': 11,
    }[x]


def pad_sequences(sequences, max_len, dtype='int32', padding='post', truncating='post', transpose=True, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[0])) * value).astype(dtype)
    mask = (np.zeros((nb_samples, max_len))).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[:, -max_len:]
        elif truncating == 'post':
            trunc = s[:, :max_len]
        if transpose:
            trunc = trunc.T
        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
            mask[idx, :len(trunc)] = 1.
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
            mask[idx, -len(trunc):] = 1.
    return x, mask



# loading the data
train_mfccs = loadmat('../data/full_tidigits_mfcc.mat')['full_tidigits_mfcc_train']
test_mfccs = loadmat('../data/full_tidigits_mfcc.mat')['full_tidigits_mfcc_test']


# all digits are in train_mfccs['mfccs_third'][0][0][0]
# all labels are then in train_mfccs['digits'][0][0][0]
# but they are in the form seqA/seqB, we need to get rid of A/B
# by looking at them we actually see that in the train there are no
# distinction between A/B so we can take directly the whole string
# for the test we need to filter the last character

train_x = train_mfccs['mfccs_third'][0][0][0]
train_y = map(lambda g: g[0], train_mfccs['digits'][0][0][0])
test_x = test_mfccs['mfccs_third'][0][0][0]
test_y = map(lambda g: g[0][:-1], test_mfccs['digits'][0][0][0])


max_len_train = len(test_y[0])
min_len_train = 1e10
max_len_test = len(train_y[0])
min_len_test = 1e10

for i in test_y:
    # print "{}/{}".format(i, len(i))
    max_len_test = max(max_len_test, len(i))
    min_len_test = min(min_len_test, len(i))

for i in train_y:
    # print "{}/{}".format(i, len(i))
    max_len_train = max(max_len_train, len(i))
    min_len_train = min(min_len_train, len(i))

print "-"*17 + "Data" + "-"*17

print "Train samples: {}, min length: {}, max length, {}".format(len(train_x), min_len_train, max_len_train)
print "Test samples: {}, min length: {}, max length, {}".format(len(test_x), min_len_test, max_len_test)
print "Couple of examples of labels {} / {} / {} / {}".format(train_y[0], train_y[1], train_y[2], train_y[3])

# we need to pad the sequences, no worries we can create masks so we can learn only on the 'real' inputs
max_len = 0
max_len = 0
for i in train_x:
    max_len = max(max_len, i.shape[1])
for i in test_x:
    max_len = max(max_len, i.shape[1])

train_x, train_x_mask = pad_sequences(train_x, max_len)
test_x, test_x_mask = pad_sequences(test_x, max_len)

max_len_y = max(max_len_train, max_len_test) + 1



# ...and for the outputs, in the mean time we need to create a sequence of codes for the
train_y_mask = np.zeros((len(train_y), max_len_y))
test_y_mask = np.zeros((len(test_y), max_len_y))
new_train_y = np.zeros((len(train_y), max_len_y))
new_test_y = np.zeros((len(test_y), max_len_y))
supp_k = 0
m = 0

for i in train_y:
    train_y_mask[m, :(len(i) + 1)] = 1.
    for k in range(len(i)):
        new_train_y[m, k] = look_up(i[k][0])
        supp_k = k
    new_train_y[m, supp_k + 1] = look_up('#')
    m += 1
supp_k = 0
m = 0
for i in test_y:
    test_y_mask[m, :(len(i) + 1)] = 1.
    for k in range(len(i)):
        new_test_y[m, k] = look_up(i[k][0])
        supp_k = k
    new_test_y[m, supp_k + 1] = look_up('#')
    m += 1

test_y = new_test_y
train_y = new_train_y



print "Train_x : {}".format(train_x.shape)
print "Test_x : {}".format(test_x.shape)
print "Train_y : {}".format(new_train_y.shape)
print "Test_y : {}".format(new_test_y.shape)
print "Train_x_mask : {}".format(train_x_mask.shape)
print "Test_x_mask : {}".format(test_x_mask.shape)
print "Train_y_mask : {}".format(train_y_mask.shape)
print "Test_y_mask : {}".format(test_y_mask.shape)

# plot some examples
# plt.figure()
# plt.imshow(train_x[sample], interpolation='nearest', aspect='auto')
# plt.xlabel('MFCCs')
# plt.ylabel('Time')
# plt.title("{}".format(new_train_y[sample]))
# plt.show()





# let's create some batches for the training
def get_batch(batch_size=3):
    idx = np.random.randint(0, len(train_x), size=batch_size)
    return train_x[idx], train_y[idx], train_x_mask[idx], train_y_mask[idx]

# let us create the model, in this case we will use a simple RNN encoder-decoder with attention
# this mean that our encoder feeds directly into the LSTM decoder that at the end of the inputs creates the output

BATCH_SIZE = 100
MAX_LENGHT_INPUT = max_len
MAX_LENGHT_OUPUT = max_len_y
MFCCs = train_x.shape[2]  # should be 39
NUM_UNITS_ENC = 20
NUM_UNITS_DEC = 20
NUM_OUTS = 12  # 1-9 + Z + O + #
#print "MFCCs: {}".format(MFCCs)

#some data for debug
v_train_x, v_train_y, v_train_x_mask, v_train_y_mask = get_batch()

x_sym = T.itensor3()
x_mask_sym = T.matrix()
y_sym = T.imatrix()
y_mask_sym = T.matrix()

l_in = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT, MFCCs))
print "-"*15 + "Network" + "-"*16
print "Input Layer: {}".format(lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: v_train_x}).shape)

l_in_mask = lasagne.layers.InputLayer((BATCH_SIZE, MAX_LENGHT_INPUT))
print "Mask Input Layer: {}".format(lasagne.layers.get_output(l_in_mask, inputs={l_in_mask: x_mask_sym}).eval({x_mask_sym: v_train_x_mask}).shape)

l_enc = lasagne.layers.GRULayer(l_in, num_units=NUM_UNITS_ENC, mask_input=l_in_mask)
print "GRU Encoder: {}".format(lasagne.layers.get_output(l_enc, inputs={l_in: x_sym, l_in_mask: x_mask_sym}).eval({x_sym: v_train_x, x_mask_sym: v_train_x_mask}).shape)

l_dec = LSTMAttentionDecodeFeedbackLayer(l_enc,
                                         num_units=NUM_UNITS_DEC,
                                         aln_num_units=8,
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

loss_mean = T.mean(loss * y_mask_sym.flatten())

argmax = T.argmax(out_decoder, axis=-1)
eq = T.eq(argmax, y_sym)
acc = eq.sum() / y_mask_sym.sum()

# train
all_params = lasagne.layers.get_all_params([l_out], trainable=True)

all_grads = [T.clip(g, -3, 3) for g in T.grad(loss_mean, all_params)]
all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)

updates = lasagne.updates.adam(all_grads, all_params)

train_func = theano.function([x_sym, y_sym, x_mask_sym, y_mask_sym],
                             [loss_mean, acc, out_decoder],
                             updates=updates,
                             allow_input_downcast=True)

test_func = theano.function([x_sym, y_sym, x_mask_sym, y_mask_sym],
                            [loss_mean, acc, l_dec.alpha, out_decoder],
                            allow_input_downcast=True)


# it's the moment to train
# gen some val data

val_x, val_y, val_x_mask, val_y_mask = test_x, test_y, test_x_mask, test_y_mask

samples_2_process = 8001
samples_processed = 0
val_acc = []
val_processed = []
batch_size_train = 100
loss_print = 8000
val_step = 8000
out_out = []
loss_along = []
print "-"*15 + "Training" + "-"*15

try:
    while samples_processed < samples_2_process:
        b_train_x, b_train_y, b_train_x_mask, b_train_y_mask = get_batch(batch_size=batch_size_train)
        avg_loss, acc_train, _ = train_func(b_train_x, b_train_y, b_train_x_mask, b_train_y_mask)
        loss_along += [avg_loss]
        samples_processed += batch_size_train
        if samples_processed % loss_print is 0:
            print "Loss batch = {}".format(avg_loss)
        if samples_processed % val_step is 0:
            loss_val, curr_acc, alphas, out = test_func(val_x, val_y, val_x_mask, val_y_mask)
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
write_model_data(l_out, 'model_LSTM_attention_{}'.format(date))
savemat('out_LSTM_attention_{}'.format(date), {'out_out': out_out, 'val_acc': val_acc, 'loss_along': loss_along})


# def main():
#
#
# if __name__ == '__main__':
#     main()
