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
cor = loadmat('../data/come_done_COR.mat')['COR']
rec_cor_1 = np.zeros_like(cor)
rec_cor_2 = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq+1, d))
n_epochs = 30
mask = np.zeros((freq+1, 2))
mask2 = np.zeros((freq+1, 2))
mask[:, 0] = 1.
mask[:, 1] = 1.
all_params = None

# fake data for debugginf
x_sym = T.dmatrix()
latent_sym = T.dmatrix()
mask_sym = T.dmatrix()
x_fake = np.ones((64, d))
x_fake_2 = np.ones((32, d))

# create the model UGLY
l_in_1 = lasagne.layers.InputLayer(shape=(None, d))
l_AE_1 = AE(l_in_1, 200, batch_size=64, nonlinearity=rectify, p=0.0)

# fake_latent = l_AE_1.get_output_latent(x_sym).eval({x_sym: x_fake})
# print "1 latent {}".format(fake_latent.shape)
# print "1 recon {}".format(l_AE_1.get_complete_reconstruction(x_sym).eval({x_sym: x_fake}).shape)
l_in_2 = lasagne.layers.InputLayer(shape=(None, 200))
l_AE_2 = AE(l_in_2, 2, batch_size=32, nonlinearity=rectify)
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

for t in range(time):
    x = cor[t, :, :]
    x_prime = np.zeros((d, 1))
    for rate in range(rates/2):
        d11 = x[:, rate]
        d1 = np.outer(d11, d11)
        d22 = x[:, rates/2 + rate]
        d2 = np.outer(d22, d22)
        x_prime[rate*freq:(rate+1)*freq, 0] = np.real((d11 + d22) / 2)
        c_mat[:-1, slice(rate * freq, (rate + 1) * freq)] = np.real((d1 + d2) / 2)
    x_prime = x_prime.transpose()
    c_mat[-1, :] = x_prime

    for epoch in range(n_epochs):
        c_mat_sc = c_mat[np.random.randint(0, freq+1, size=(freq+1))]
        for i in range(8):
            eval_train = train_model_1(c_mat_sc[i*64:(1+i)*64])
    print "Layer 1 %.10f (time=%i)" % (eval_train, t)

    z = l_AE_1.get_output_latent(x_sym, deterministic=True).eval({x_sym: c_mat})

    for epoch in range(100):
        z_sc = z[np.random.randint(0, freq+1, size=(freq+1))]
        for i in range(16):
            eval_train = train_model_2(z_sc[i*32:(1+i)*32])
    print "Layer 1 %.10f (time=%i)" % (eval_train, t)

    # z_prime = l_AE_1.get_output_latent(x_sym).eval({x_sym: x_prime})
    # half_way_1 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z_prime, mask_sym: mask})
    # half_way_2 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z_prime, mask_sym: mask2})
    # x_recon_1 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_1})
    # x_recon_2 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_2})

    #err = (x_prime - x_recon_1)**2
    # let's do it step by step

    enc1 = l_AE_1.get_output_latent(x_sym, deterministic=True).eval({x_sym: c_mat})
    enc2 = l_AE_2.get_output_latent(x_sym, deterministic=True).eval({x_sym: enc1})
    dec1 = l_AE_2.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: enc2})
    dec2 = l_AE_1.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: dec1})

    dec_from_enc1 = l_AE_1.get_reconstruction_from_latent(x_sym, deterministic=True).eval({x_sym: enc1})

    # c_comp = l_AE_1.get_complete_reconstruction(x_sym).eval({x_sym: c_mat})
    # z = l_AE_1.get_output_latent(x_sym).eval({x_sym: c_mat})
    # half_way = l_AE_2.get_complete_reconstruction(x_sym).eval({x_sym: z})
    # half_way_1 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z, mask_sym: mask})
    # half_way_2 = l_AE_2.get_complete_reconstruction(x_sym, mask_sym).eval({x_sym: z, mask_sym: mask2})
    # c_recon_1 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_1})
    # c_recon_2 = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way_2})
    # c_recon = l_AE_1.get_reconstruction_from_latent(x_sym).eval({x_sym: half_way})

    plt.figure
    plt.subplot(2, 3, 1)
    plt.imshow(c_mat, interpolation='nearest', aspect='auto')
    plt.subplot(2, 3, 2)
    plt.imshow(dec_from_enc1, interpolation='nearest', aspect='auto')
    plt.subplot(2, 3, 3)
    # print dec_from_enc1
    plt.imshow(dec2, interpolation='nearest', aspect='auto')
    plt.subplot(2, 3, 4)
    # print dec2
    plt.imshow(z, interpolation='nearest', aspect='auto')
    plt.subplot(2, 3, 5)
    plt.imshow(dec1, interpolation='nearest', aspect='auto')
    rec_cor_1[t, :, :rates/2] = dec2[-1].reshape(freq, rates/2)
    rec_cor_2[t, :, :rates/2] = dec2[-1].reshape(freq, rates/2)
    tx = np.arange(d)
    plt.subplot(2, 3, 6)
    plt.plot(tx, dec2[-1], 'r',  tx, dec2[-1], 'g', tx, c_mat[- 1], 'b')
    plt.show()
savemat('come_done_filtered.mat', {'rec_cor_1': rec_cor_1, 'rec_cor_2': rec_cor_2})