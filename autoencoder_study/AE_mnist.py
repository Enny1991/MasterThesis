import lasagne
import theano
import theano.tensor as T
import numpy as np
import time as tm
from lasagne.nonlinearities import rectify,identity
from scipy.io import loadmat, savemat
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import init
from lasagne.updates import adam, sgd, rmsprop
from matplotlib import pyplot as plt


class AE(Layer):

    def __init__(self, incoming, latent, nonlinearity=None,
                 W=init.Uniform(),
                 b=init.Uniform(),
                 batch_size=512,
                 p=0.0,
                 **kwargs):
        super(AE, self).__init__(incoming, **kwargs)
        self.num_batch, self.num_units = self.input_shape
        if nonlinearity is None:
            self.nonlinearity = identity
        else:
            self.nonlinearity = nonlinearity

        self.n_hidden = latent
        self.x = incoming
        self.batch_size = batch_size
        #num_inputs = int(np.prod(self.input_shape[1:]))
        rng = np.random.RandomState(123)
        self.drng = rng
        self.rng = RandomStreams(rng.randint(2 ** 30))
        self.p = p

        initial_W = np.asarray(
            rng.uniform(
                    low=-4 * np.sqrt(6. / (self.n_hidden + self.num_units)),
                    high=4 * np.sqrt(6. / (self.n_hidden + self.num_units)),
                    size=(self.num_units, self.n_hidden)
            ),
            dtype=theano.config.floatX
        )

        #self.W = self.create_param(initial_W, (num_inputs, n_hidden), name="W")
        #self.bvis = self.create_param(bvis, (num_units,), name="bvis") if bvis is not None else None
        #self.bhid = self.create_param(bhid, (n_hidden,), name="bhid") if bhid is not None else None
        self.W = theano.shared(value=initial_W, name='W', borrow=True)

        bvis = theano.shared(
                value=np.zeros(
                    self.num_units,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        bhid = theano.shared(
                value=np.zeros(
                    self.n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = theano.shared(value=initial_W.T, name='W_T', borrow=True)

    def get_params(self):
        #return [self.W] + self.get_bias_params()
        return [self.W_prime] + [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b, self.b_prime] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (self.batch_size,self.n_hidden)

    def get_output_latent(self, x, deterministic=False):
        # dropout
        retain_prob = 1 - self.p
        if not deterministic:
            return self.nonlinearity(T.dot(x, self.W) + self.b) * self.rng.binomial((self.batch_size, self.n_hidden),
                                                                                    p=retain_prob,
                                                                                    dtype=theano.config.floatX)
        else:
            return self.nonlinearity(T.dot(x, self.W) + self.b)

    def get_corrupted_output_latent(self, x, corruption=0.0):
        corr = self.rng.normal(size=x.shape, avg=0, std=corruption)
        return self.nonlinearity(T.dot(x, self.W) + self.b + corr)

    def get_complete_reconstruction(self, x, mask=None):
        if mask is not None:
            latent = self.get_output_latent(x) * mask
        else:
            latent = self.get_output_latent(x)
        #return T.dot(latent, self.W_prime) + self.b_prime
        return self.get_reconstruction_from_latent(latent)

    def get_reconstruction_from_latent(self, z, deterministic=False):
        # dropout
        return T.dot(z, self.W_prime) + self.b_prime

    def get_output_for(self, input, **kwargs):
        z = self.get_output_latent(input)
        x_tilde = self.get_reconstruction_from_latent(z)
        return x_tilde

    def build_model(self, update, update_args):
        self.update = update
        self.update_args = update_args
        x = T.dmatrix()
        z = self.get_output_latent(x)
        loss_eval = (lasagne.objectives.squared_error(self.get_reconstruction_from_latent(z), x)).sum()
        loss_eval /= 2 * self.batch_size

        all_params = self.get_params()
        updates = self.update(loss_eval, all_params, update_args)

        train_model = theano.function([x], loss_eval, updates=updates)

        #test_model = theano.function([self.test_x], loss_eval,
        #                             givens={self.x: self.test_x[self.batch_slice], },)

        #validate_model = theano.function([self.index], loss_eval,
        #                                 givens={self.x: self.validation_x[self.batch_slice], },)

        return train_model






# from here we apply our data

d = 784

n_epochs = 100

# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((500, d))
x_fake_2 = np.ones((500, d))

# create the model UGLY
l_in_1 = lasagne.layers.InputLayer(shape=(None, d))
l_AE_1 = AE(l_in_1, 200, batch_size=500, nonlinearity=rectify, p=0.0)

# fake_latent = l_AE_1.get_output_latent(x_sym).eval({x_sym: x_fake})
# print "1 latent {}".format(fake_latent.shape)
# print "1 recon {}".format(l_AE_1.get_complete_reconstruction(x_sym).eval({x_sym: x_fake}).shape)
l_in_2 = lasagne.layers.InputLayer(shape=(None, 200))
l_AE_2 = AE(l_in_2, 2, batch_size=500, nonlinearity=rectify)
#
# fake_latent_2 = l_AE_2.get_output_latent(latent_sym).eval({latent_sym: fake_latent})
# print "1 latent {}".format(fake_latent_2.shape)
# print "1 recon {}".format(l_AE_2.get_complete_reconstruction(latent_sym).eval({latent_sym: fake_latent}).shape)
#
# # complete pipeline
# rec_first_hid = l_AE_2.get_reconstruction_from_latent(latent_sym).eval({latent_sym: fake_latent_2})
# recon = l_AE_1.get_reconstruction_from_latent(latent_sym).eval({latent_sym: rec_first_hid})
#
# print "rec inside {}".format(rec_first_hid.shape)
# print "rec outside {}".format(recon.shape)

train_model_1 = l_AE_1.build_model(adam, 1e-3)
train_model_2 = l_AE_2.build_model(adam, 1e-3)

def load_mnist():

    data = np.load('../data/mnist.npz')
    num_classes = 10
    x_train, targets_train = data['X_train'].astype('float32'), data['y_train']
    x_valid, targets_valid = data['X_valid'].astype('float32'), data['y_valid']
    x_test, targets_test = data['X_test'].astype('float32'), data['y_test']


    return (x_train, targets_train), (x_test, targets_test), (x_valid, targets_valid)


(train_x, train_t), (test_x, test_t), (valid_x, valid_t) = load_mnist()
total = train_x.shape[0]

for epoch in range(n_epochs):
    for i in range(total/500):
        t_batch = train_x[i*500:(i+1)*500]
        eval_train = train_model_1(t_batch)
    print "Layer 1 %.10f (time=%i)" % (eval_train,epoch)

z = l_AE_1.get_output_latent(x_sym, deterministic=True).eval({x_sym: test_x})

# for epoch in range(1):
#     for i in range(total/500):
#         t_batch = z[i*500:(i+1)*500]
#         eval_train = train_model_2(t_batch)
#     print "Layer 2 %.10f (time=%i)" % (eval_train,epoch)

    # z_prime = l_AE_1.get_output_latent(x_sym).eval({x_sym: x_prime})
    # half_way_1 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z_prime, mask_sym: mask})
    # half_way_2 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z_prime, mask_sym: mask2})
    # x_recon_1 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_1})
    # x_recon_2 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_2})

    #err = (x_prime - x_recon_1)**2
    # let's do it step by step

enc1 = l_AE_1.get_output_latent(x_sym, deterministic=True).eval({x_sym: test_x})
enc2 = l_AE_2.get_output_latent(x_sym, deterministic=True).eval({x_sym: enc1})
dec1 = l_AE_2.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: enc2})
dec2 = l_AE_1.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: z})

dec_from_enc1 = l_AE_1.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: enc1})



print dec2.shape
print test_x.shape
fig = plt.figure()
i = 0
test_x_eval = test_x
subset = np.random.randint(0, len(test_x_eval), size=50)
img_out = np.zeros((28 * 2, 28 * len(subset)))
x = np.array(test_x_eval)[np.array(subset)]
for y in range(len(subset)):
    x_a, x_b = 0 * 28, 1 * 28
    x_recon_a, x_recon_b = 1 * 28, 2 * 28
    ya, yb = y * 28, (y + 1) * 28
    im = np.reshape(x[i], (28, 28))
    im_recon = np.reshape(dec2[i], (28, 28))
    img_out[x_a:x_b, ya:yb] = im
    img_out[x_recon_a:x_recon_b, ya:yb] = im_recon
    i += 1
m = plt.matshow(img_out, cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.show()