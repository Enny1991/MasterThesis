import lasagne
import theano
import theano.tensor as T
import numpy as np
from IPython import display
from matplotlib import pyplot as plt


def look_up_numbers(x):
    return{
        0: 'zero ',
        1: 'one ',
        2: 'two ',
        3: 'three ',
        4: 'four ',
        5: 'five ',
        6: 'six ',
        7: 'seven ',
        8: 'eight ',
        9: 'nine ',
    }[x]


def look_up_words(x):
    return{
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
    }[x]


def look_up(x):
    return{
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '#': 10,
        ' ': 11,
        'e': 12,
        'g': 13,
        'f': 14,
        'i': 15,
        'h': 16,
        'o': 17,
        'n': 18,
        's': 19,
        'r': 20,
        'u': 21,
        't': 22,
        'w': 23,
        'v': 24,
        'x': 25,
        'z': 26,
    }[x]


def get_batch(batch_size=10, max_digits=2, min_digits=1):
    rand_in = []

    for i in range(batch_size):
        t = tuple()
        for j in range(np.random.randint(min_digits, max_digits+1)):
            t = t + (np.random.randint(0, 10),)
        rand_in.append(t)

    inputs = np.zeros((batch_size, 6 * max_digits))
    inputs_mask = np.zeros((batch_size, 6 * max_digits))
    targets = np.zeros((batch_size, max_digits + 1))
    target_masks = np.zeros((batch_size, max_digits + 1))
    text_inputs = []
    text_targets = []
    k = 0
    for tup in rand_in:
        text_inputs.append([])
        text_targets.append([])
        for num in range(len(tup)):
            text_inputs[k] += (look_up_numbers(tup[num]))
            text_targets[k] += (look_up_words(tup[num]))
        text_inputs[k] = text_inputs[k][:-1]
        text_targets[k] += '#'
        m = 0
        for char in text_inputs[k]:
            inputs[k, m] = look_up(char)
            inputs_mask[k, m] = 1.
            m += 1
        m = 0
        for char in text_targets[k]:
            targets[k, m] = look_up(char)
            target_masks[k, m] = 1.
            m += 1
        k += 1
    return inputs.astype('int32'), \
           inputs_mask.astype('float32'), \
           targets.astype('int32'), \
           target_masks.astype('float32'), \
           text_inputs, \
           text_targets


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


# batch_size = 3
# inputs, input_masks, targets, target_masks, text_inputs, text_targets = \
#      get_batch(batch_size=batch_size, max_digits=10, min_digits=2)
#
# for i in range(batch_size):
#     print "\nSAMPLE",i
#     print "TEXT INPUTS:\t\t", text_inputs[i]
#     print "TEXT TARGETS:\t\t", text_targets[i]
#     print "ENCODED INPUTS:\t\t", inputs[i]
#     print "MASK INPUTS:\t\t", input_masks[i]
#     print "ENCODED TARGETS:\t", targets[i]
#     print "MASK TARGETS:\t\t", target_masks[i]

BATCH_SIZE = 100
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = 10
MAX_DIGITS = 20
MIN_DIGITS = 2 #currently only support for same length outputs - we'll leave it for an exercise to add support for varying length targets
NUM_INPUTS = 27
NUM_OUTPUTS = 11 #(0-9 + '#')
batch_size = 3
inputs, input_masks, targets, target_masks, text_inputs, text_targets = \
     get_batch(batch_size=BATCH_SIZE, max_digits=MAX_DIGITS, min_digits=MIN_DIGITS)

#symbolic theano variables. Note that we are using imatrix for X since it goes into the embedding layer
x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()
ymask_sym = T.matrix()


X = np.random.randint(0,10,size=(BATCH_SIZE,MIN_DIGITS)).astype('int32')
Xmask = np.ones((BATCH_SIZE,MIN_DIGITS)).astype('float32')

##### ENCODER START #####
l_in = lasagne.layers.InputLayer((None, None))
l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS,
                                      W=np.eye(NUM_INPUTS, dtype='float32'),
                                      name='Embedding')

l_emb.params[l_emb.W].remove('trainable')

l_mask_enc = lasagne.layers.InputLayer((None, None))
l_enc = lasagne.layers.GRULayer(l_emb, num_units=NUM_UNITS_ENC, name='GRUEncoder', mask_input=l_mask_enc)


# slice last index of dimension 1
l_last_hid = lasagne.layers.SliceLayer(l_enc, indices=-1, axis=1)

# if you would like to feed the decoder with more steps it is necessary to retrieve the all output and take n steps
# needed and use that as a new input for the decoder
# instead of repeat layer you would need to create your new input layer

##### START OF DECODER######
l_in_rep = RepeatLayer(l_last_hid, n=MAX_DIGITS+1) #we add one to allow space for the end of sequence character


l_dec = lasagne.layers.GRULayer(l_in_rep, num_units=NUM_UNITS_DEC, name='GRUDecoder')


# We need to do some reshape voodo to connect a softmax layer to the decoder.
# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples
# In short this line changes the shape from
# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units).
# We need to do this since the softmax is applied to the last dimension and we want to
# softmax the output at each position individually
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))


l_softmax = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      name='SoftmaxOutput')

# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing
#us to use different batch sizes in the model.
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))

###END OF DECODER######

output_decoder_train = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym},
                                                deterministic=False)


#cost function
out_mask = (T.reshape(T.extra_ops.repeat(ymask_sym.flatten(), NUM_OUTPUTS, axis=0),((MAX_DIGITS+1)*BATCH_SIZE,NUM_OUTPUTS)))
total_cost = T.nnet.categorical_crossentropy(
    T.reshape(output_decoder_train, (-1, NUM_OUTPUTS)), y_sym.flatten())

OO =  lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
     {x_sym: inputs, xmask_sym: input_masks})
#
# m = total_cost.eval({output_decoder_train: OO, y_sym: targets})
# f = T.mean(m * ymask_sym.flatten())
# print f.eval({ymask_sym: target_masks})

mean_cost = T.mean(total_cost * ymask_sym.flatten())


#accuracy function
argmax = T.argmax(output_decoder_train, axis=-1)
eq = T.eq(argmax, y_sym)
acc = eq.sum() / ymask_sym.sum()  # gives float64 because eq is uint8, T.cast(eq, 'float32') will fix that...

#print acc.eval({y_sym: targets, ymask_sym: target_masks})
#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params([l_out], trainable=True)



#add grad clipping to avoid exploding gradients
all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)


updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

train_func = theano.function([x_sym, y_sym, xmask_sym, ymask_sym], [mean_cost, acc, output_decoder_train], updates=updates, allow_input_downcast=True, on_unused_input='warn')

test_func = theano.function([x_sym, y_sym, xmask_sym, ymask_sym], [acc, output_decoder_train], allow_input_downcast=True, on_unused_input='warn')


#Generate some validation data
Xval, Xmask_val, Yval, Ymask_val, text_inputs_val, text_targets_val = \
    get_batch(batch_size=5000, max_digits=MAX_DIGITS, min_digits=MIN_DIGITS)

val_interval = 5000
samples_to_process = 3e6
samples_processed = 0

val_samples = []
costs, accs = [], []
plt.figure()
try:
    while samples_processed < samples_to_process:
        inputs, input_masks, targets, target_masks, _, _ = \
            get_batch(batch_size=BATCH_SIZE, max_digits=MAX_DIGITS,min_digits=MIN_DIGITS)
        batch_cost, batch_acc, batch_output = train_func(inputs, targets, input_masks, target_masks)
        costs += [batch_cost]
        samples_processed += BATCH_SIZE
        #validation data
        if samples_processed % val_interval == 0:
            #print "validating"
            val_acc, val_output = test_func(Xval, Yval, Xmask_val, Ymask_val)
            val_samples += [samples_processed]
            accs += [val_acc]
            plt.plot(val_samples, accs)
            plt.ylabel('Validation Accuracy', fontsize=15)
            plt.xlabel('Processed samples', fontsize=15)
            plt.title('', fontsize=20)
            plt.grid('on')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
except KeyboardInterrupt:
    pass