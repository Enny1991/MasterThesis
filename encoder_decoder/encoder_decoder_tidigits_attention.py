import theano.tensor as T
import theano
import lasagne
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

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

def pad_sequences(sequences, max_len, dtype='int32', padding='pre', truncating='pre', transpose=True, value=0.):
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len, sequences[0].shape[0])) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[:, -max_len:]
        elif truncating == 'post':
            trunc = s[:, :max_len]
        if transpose:
            trunc = trunc.T
        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
    return x

# loading the data
train_mfccs = loadmat('../data/full_tidigits_mfcc.mat')['full_tidigits_mfcc_train']
test_mfccs = loadmat('../data/full_tidigits_mfcc.mat')['full_tidigits_mfcc_test']


# all digits are in train_mfccs['mfccs_third'][0][0][0]
# all labels are then in train_mfccs['digits'][0][0][0]
# but they are in the form seqA/seqB, we need to get rid of A/B

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

print "Train samples: {}, min length: {}, max length, {}".format(len(train_x), min_len_train, max_len_train)
print "Test samples: {}, min length: {}, max length, {}".format(len(test_x), min_len_test, max_len_test)
print "couple of examples of labels {} / {} / {} / {}".format(train_y[0], train_y[1], train_y[2], train_y[3])

# we need to pad the sequences, no worries we can create masks so we can learn only on the 'real' inputs
max_len = 0
max_len = 0
for i in train_x:
    max_len = max(max_len, i.shape[1])
for i in test_x:
    max_len = max(max_len, i.shape[1])

train_x = pad_sequences(train_x, max_len)
test_x = pad_sequences(test_x, max_len)

# need to create masks for inputs...
train_x_mask = np.zeros((len(train_x), max_len))
test_x_mask = np.zeros((len(test_x), max_len))
for i in train_x:
    train_x_mask[:len(i)] = 1.
for i in test_x:
    test_x_mask[:len(i)] = 1.

# ...and for the outputs, in the mean time we need to create a sequence of codes for the
max_len_y = max(max_len_train, max_len_test) + 1
train_y_mask = np.zeros((len(train_y), max_len_y))
test_y_mask = np.zeros((len(test_y), max_len_y))
new_train_y = np.zeros((len(train_y), max_len_y))
new_test_y = np.zeros((len(test_y), max_len_y))
supp_k = 0
m = 0
for i in train_y:
    train_y_mask[:(len(i) + 1)] = 1.
    for k in range(len(i)):
        new_train_y[m, k] = look_up(i[k][0])
        supp_k = k
    new_train_y[m, supp_k] = look_up('#')
    m += 1
supp_k = 0
m = 0
for i in test_y:
    test_y_mask[:(len(i) + 1)] = 1.
    for k in range(len(i)):
        new_test_y[m, k] = look_up(i[k][0])
        supp_k = k
    new_test_y[m, supp_k] = look_up('#')
    m += 1


print "train_x shape: {}".format(train_x.shape)
print "test_x shape: {}".format(test_x.shape)
print "new_train_y shape: {}".format(new_train_y.shape)
print "new_test_y shape: {}".format(new_test_y.shape)
print "train_x_mask shape: {}".format(train_x_mask.shape)
print "test_x_mask shape: {}".format(test_x_mask.shape)
print "train_y_mask shape: {}".format(train_y_mask.shape)
print "test_y_mask shape: {}".format(test_y_mask.shape)


# let's create some batches for the training


