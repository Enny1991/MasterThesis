from decoder_attention import LSTMAttentionDecodeFeedbackLayer
import theano
import theano.tensor as T
import lasagne
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
# you can acces the attetion weights alpha by adding l_dec.alpha
# to the output variables in the theano function

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


BATCH_SIZE = 100
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = 10
MAX_DIGITS = 20
MIN_DIGITS = MAX_DIGITS #currently only support for same length outputs - we'll leave it for an exercise to add support for varying length targets
NUM_INPUTS = 27
NUM_OUTPUTS = 11 #(0-9 + '#')


x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()


#dummy data to test implementation
X = np.random.randint(0,10,size=(BATCH_SIZE,15)).astype('int32')
Xmask = np.ones((BATCH_SIZE,NUM_INPUTS)).astype('float32')

l_in = lasagne.layers.InputLayer((None, None))
l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS,
                                      W=np.eye(NUM_INPUTS,dtype='float32'),
                                      name='Embedding')
##### ENCODER START #####
l_in = lasagne.layers.InputLayer((None, None))
l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS,
                                      W=np.eye(NUM_INPUTS,dtype='float32'),
                                      name='Embedding')
#Here we'll remove the trainable parameters from the embeding layer to constrain
#it to a simple "one-hot-encoding". You can experiment with removing this line
l_emb.params[l_emb.W].remove('trainable')
print lasagne.layers.get_output(l_emb, inputs={l_in: x_sym}).eval(
    {x_sym: X}).shape
T.grad(lasagne.layers.get_output(l_emb, inputs={l_in: x_sym}).sum(),
       lasagne.layers.get_all_params(l_emb, trainable=True))




l_mask_enc = lasagne.layers.InputLayer((None, None))
l_enc = lasagne.layers.GRULayer(l_emb, num_units=NUM_UNITS_ENC, name='GRUEncoder', mask_input=l_mask_enc)
print lasagne.layers.get_output(l_enc, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape
T.grad(lasagne.layers.get_output(l_enc, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).sum(),
       lasagne.layers.get_all_params(l_enc, trainable=True))
####END OF ENCODER######


####START OF DECODER######
#note that the decoder have its own input layer, we'll use that to plug in the output
#from the encoder later
l_dec = LSTMAttentionDecodeFeedbackLayer(l_enc,
                                        num_units=NUM_UNITS_DEC,
                                        aln_num_units=20,
                                        n_decodesteps=MAX_DIGITS+1,
                                        name='LSTMDecoder')
print lasagne.layers.get_output(l_dec, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape
T.grad(lasagne.layers.get_output(l_dec, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).sum(),
       lasagne.layers.get_all_params(l_dec, trainable=True))

# We need to do some reshape voodo to connect a softmax layer to the decoder.
# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))
l_softmax = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      name='SoftmaxOutput')
# print lasagne.layers.get_output(l_softmax, x_sym).eval({x_sym: X}).shape
# reshape back to 3d format (here we tied the batch size to the shape of the symbolic variable for X allowing
#us to use different batch sizes in the model)
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
print lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym}, deterministic=False).eval(
    {x_sym: X, xmask_sym: Xmask}).shape
T.grad(lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).sum(),
       lasagne.layers.get_all_params(l_dec, trainable=True))

print ""
###END OF DECODER######

#get output of encoder using X and Xmask as input
output_decoder_train = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym},
                                                 deterministic=False)

#cost function
total_cost = T.nnet.categorical_crossentropy(
    T.reshape(output_decoder_train, (-1, NUM_OUTPUTS)), y_sym.flatten())
mean_cost = T.mean(total_cost)
#accuracy function
acc = T.mean(T.eq(T.argmax(output_decoder_train,axis=-1),y_sym))

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params(l_out, trainable=True)

print "Trainable Model Parameters"
print "-"*40
for param in all_parameters:
    print param, param.get_value().shape
print "-"*40

#add grad clipping to avoid exploding gradients
all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

#Compile Theano functions
updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)
train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc, output_decoder_train], updates=updates)
#since we don't have any stochasticity in the network we will just use the training graph without any updates given
test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_decoder_train, l_dec.alpha])

#Generate some validation data
Xval, Xmask_val, Yval, Ymask_val, text_inputs_val, text_targets_val = \
    get_batch(batch_size=5000, max_digits=MAX_DIGITS, min_digits=MIN_DIGITS)

val_interval = 5000
samples_to_process = 1.5e5
samples_processed = 0
val_samples = []
costs, accs = [], []
plt.figure()
try:
    while samples_processed < samples_to_process:
        inputs, input_masks, targets, target_masks, _, _ = \
            get_batch(batch_size=BATCH_SIZE, max_digits=MAX_DIGITS, min_digits=MIN_DIGITS)
        batch_cost, batch_acc, batch_output = train_func(inputs, targets, input_masks)
        costs += [batch_cost]
        samples_processed += BATCH_SIZE
        #print i, samples_processed
        #validation data
        if samples_processed % val_interval == 0:
            #print "validating"
            val_acc, val_output, alpha = test_func(Xval, Yval, Xmask_val)
            val_samples += [samples_processed]
            accs += [val_acc]
            plt.plot(val_samples,accs)
            plt.ylabel('', fontsize=15)
            plt.xlabel('Processed samples', fontsize=15)
            plt.title('Validation Accuracy', fontsize=20)
            plt.grid('on')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
except KeyboardInterrupt:
    pass
