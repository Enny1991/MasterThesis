import theano
import numpy as np
import theano.tensor as T
import lasagne
from scipy.io import loadmat, savemat
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer, Layer
from lasagne.nonlinearities import linear, rectify, identity
from lasagne.objectives import squared_error
from lasagne.layers import get_all_layers, get_output, get_all_params, InputLayer
from lasagne.updates import adam
from matplotlib import pyplot as plt
from lasagne import init
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import math
import time as tm

theano.config.floatX = 'float64'
def _shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y


def generate_synthetic_data(dat_size=1.e4):
    rng = np.random.RandomState(42)
    v = rng.normal(0, .02, size=dat_size).reshape((dat_size, 1))
    x = rng.uniform(0., 1., size=dat_size).reshape((dat_size, 1))
    y = x + 0.3 * np.sin(2 * np.pi * (x + v)) + 0.3 * np.sin(4*np.pi*(x + v)) + v

    train_x = x[:dat_size/2]
    train_y = y[:dat_size/2]

    test_x = x[dat_size/2 + 1:]
    test_y = y[dat_size/2 + 1:]

    return _shared_dataset((train_x, train_y)), _shared_dataset((test_x, test_y))

((train_x, train_y), (test_x, test_y)) = generate_synthetic_data(1e4)

test_x_unshared = test_x.eval()
test_y_unshared = test_y.eval()

c = - 0.5 * math.log(2*math.pi)


def normal(x, mean, sd):
    return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)


def normal2(x, mean, logvar):
    return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))


def standard_normal(x):
    return c - x**2 / 2


class VAELayer(Layer):

    def __init__(self, incoming, encoder, decoder,
                 x_distribution='gaussian',
                 pz_distribution='gaussian',
                 qz_distribution='gaussian',
                 latent_size=50,
                 W=init.Normal(0.01),
                 b=init.Normal(0.01),
                 **kwargs):
        super(VAELayer, self).__init__(incoming, **kwargs)
        num_batch, n_features = self.input_shape
        self.num_batch = num_batch
        self.n_features = n_features
        self.x_distribution = x_distribution
        self.pz_distribution = pz_distribution
        self.qz_distribution = qz_distribution
        self.encoder = encoder
        self.decoder = decoder
        self._srng = RandomStreams()

        if self.x_distribution not in ['gaussian', 'bernoulli']:
            raise NotImplementedError
        if self.pz_distribution not in ['gaussian', 'gaussianmarg']:
            raise NotImplementedError
        if self.qz_distribution not in ['gaussian', 'gaussianmarg']:
            raise NotImplementedError

        self.params_encoder = lasagne.layers.get_all_params(encoder)
        self.params_decoder = lasagne.layers.get_all_params(decoder)
        for p in self.params_encoder:
            p.name = "VAELayer encoder :" + p.name
        for p in self.params_decoder:
            p.name = "VAELayer decoder :" + p.name

        self.num_hid_enc = encoder.output_shape[1]
        self.num_hid_dec = decoder.output_shape[1]
        self.latent_size = latent_size

        self.W_enc_to_z_mu = self.add_param(W, (self.num_hid_enc, latent_size))
        self.b_enc_to_z_mu = self.add_param(b, (latent_size,))

        self.W_enc_to_z_logsigma = self.add_param(W, (self.num_hid_enc, self.latent_size))
        self.b_enc_to_z_logsigma = self.add_param(b, (latent_size,))

        self.W_dec_to_x_mu = self.add_param(W, (self.num_hid_dec, self.n_features))
        self.b_dec_to_x_mu = self.add_param(b, (self.n_features,))

        self.W_params = [self.W_enc_to_z_mu,
                         self.W_enc_to_z_logsigma,
                         self.W_dec_to_x_mu] + self.params_encoder + self.params_decoder
        self.bias_params = [self.b_enc_to_z_mu,
                            self.b_enc_to_z_logsigma,
                            self.b_dec_to_x_mu]

        params_tmp = []
        if self.x_distribution == 'gaussian':
            self.W_dec_to_x_logsigma = self.add_param(W, (self.num_hid_dec, self.n_features))
            self.b_dec_to_x_logsigma = self.add_param(b, (self.n_features,))
            self.W_params += [self.W_dec_to_x_logsigma]
            self.bias_params += [self.b_dec_to_x_logsigma]
            self.W_dec_to_x_logsigma.name = "VAE: W_dec_to_x_logsigma"
            self.b_dec_to_x_logsigma.name = "VAE: b_dec_to_x_logsigma"
            params_tmp = [self.W_dec_to_x_logsigma, self.b_dec_to_x_logsigma]

        self.params = self.params_encoder + [self.W_enc_to_z_mu,
                                             self.b_enc_to_z_mu,
                                             self.W_enc_to_z_logsigma,
                                             self.b_enc_to_z_logsigma] + self.params_decoder + \
                                            [self.W_dec_to_x_mu, self.b_dec_to_x_mu] + params_tmp

        self.W_enc_to_z_mu.name = "VAELayer: W_enc_to_z_mu"
        self.W_enc_to_z_logsigma.name = "VAELayer: W_enc_to_z_logsigma"
        self.W_dec_to_x_mu.name = "VAELayer: W_dec_to_x_mu"
        self.b_enc_to_z_mu.name = "VAELayer: b_enc_to_z_mu"
        self.b_enc_to_z_logsigma.name = "VAELayer: b_enc_to_z_logsigma"
        self.b_dec_to_x_mu.name = "VAELayer: b_dec_to_x_mu"

    def get_params(self):
        return self.params

    def get_output_shape_for(self, input_shape):
        dec_out_shp = self.decoder.get_output_shape_for(
            (self.num_batch, self.num_hid_dec))
        if self.x_distribution == 'bernoulli':
            return dec_out_shp
        elif self.x_distribution == 'gaussian':
            return [dec_out_shp, dec_out_shp]

    def _encoder_output(self, x, *args, **kwargs):
        return lasagne.layers.get_output(self.encoder, x, **kwargs)

    def decoder_output(self, z, *args, **kwargs):
        h_decoder = lasagne.layers.get_output(self.decoder, z, **kwargs)
        if self.x_distribution == 'gaussian':
            mu_decoder = T.dot(h_decoder, self.W_dec_to_x_mu) + self.b_dec_to_x_mu
            log_sigma_decoder = T.dot(h_decoder, self.W_dec_to_x_logsigma) + self.b_dec_to_x_logsigma
            decoder_out = mu_decoder, log_sigma_decoder
        elif self.x_distribution == 'bernoulli':
            # TODO: Finish writing the output of the decoder for a bernoulli distributed x.
            decoder_out = T.nnet.sigmoid(T.dot(h_decoder, self.W_dec_to_x_mu) + self.b_dec_to_x_mu)
        else:
            raise NotImplementedError
        return decoder_out

    def get_z_mu_sigma(self, x, *args, **kwargs):
        h_encoder = self._encoder_output(x, *args, **kwargs)
        mu_encoder = T.dot(h_encoder, self.W_enc_to_z_mu) + self.b_enc_to_z_mu
        log_sigma_encoder = (T.dot(h_encoder, self.W_enc_to_z_logsigma) +
                             self.b_enc_to_z_logsigma)
        eps = self._srng.normal(log_sigma_encoder.shape)
        # TODO: Calculate the sampled z.
        z = mu_encoder + T.exp(0.5 * log_sigma_encoder) * eps
        return z, mu_encoder, log_sigma_encoder

    def get_log_distributions(self, x, *args, **kwargs):
        # sample z from q(z|x).
        h_encoder = self._encoder_output(x, *args, **kwargs)
        mu_encoder = T.dot(h_encoder, self.W_enc_to_z_mu) + self.b_enc_to_z_mu
        log_sigma_encoder = (T.dot(h_encoder, self.W_enc_to_z_logsigma) +
                             self.b_enc_to_z_logsigma)
        eps = self._srng.normal(log_sigma_encoder.shape)
        z = mu_encoder + T.exp(0.5 * log_sigma_encoder) * eps

        # forward pass z through decoder to generate p(x|z).
        decoder_out = self.decoder_output(z, *args, **kwargs)
        if self.x_distribution == 'bernoulli':
            x_mu = decoder_out
            log_px_given_z = -T.nnet.binary_crossentropy(x_mu, x)
        elif self.x_distribution == 'gaussian':
            x_mu, x_logsigma = decoder_out
            log_px_given_z = normal2(x, x_mu, x_logsigma)

        # sample prior distribution p(z).
        if self.pz_distribution == 'gaussian':
            log_pz = standard_normal(z)
        elif self.pz_distribution == 'gaussianmarg':
            log_pz = -0.5 * (T.log(2 * np.pi) + (T.sqr(mu_encoder) + T.exp(log_sigma_encoder)))

        # variational approximation distribution q(z|x)
        if self.qz_distribution == 'gaussian':
            log_qz_given_x = normal2(z, mu_encoder, log_sigma_encoder)
        elif self.qz_distribution == 'gaussianmarg':
            log_qz_given_x = - 0.5 * (T.log(2 * np.pi) + 1 + log_sigma_encoder)

        # sum over dim 1 to get shape (,batch_size)
        log_px_given_z = log_px_given_z.sum(axis=1, dtype=theano.config.floatX)  # sum over x
        log_pz = log_pz.sum(axis=1, dtype=theano.config.floatX)  # sum over latent vars
        log_qz_given_x = log_qz_given_x.sum(axis=1, dtype=theano.config.floatX)  # sum over latent vars

        return log_pz, log_qz_given_x, log_px_given_z

    def draw_sample(self, z=None, *args, **kwargs):
        if z is None:  # draw random z
            z = self._srng.normal((self.num_batch, self.latent_size))
        return self.decoder_output(z, *args, **kwargs)


class VAE:
    def __init__(self, n_in, n_hidden, n_out,
                 n_hidden_decoder=None,
                 trans_func=rectify, batch_size=513):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.l_in = InputLayer((batch_size, n_in))
        self.batch_size = batch_size
        self.transf = trans_func

        self.srng = RandomStreams()

        l_in_encoder = lasagne.layers.InputLayer(shape=(batch_size, n_in))
        l_in_decoder = lasagne.layers.InputLayer(shape=(batch_size, n_out))

        l_prev_encoder = l_in_encoder
        l_prev_decoder = l_in_decoder

        for i in range(len(n_hidden)):
            l_tmp_encoder = lasagne.layers.DenseLayer(l_prev_encoder,
                                                      num_units=n_hidden[i],
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=self.transf)
            l_prev_encoder = l_tmp_encoder

        # cause you might want a decoder which is not the mirror of the encoder
        if n_hidden_decoder is None:
            n_hidden_decoder = n_hidden
        self.n_hidden_decoder = n_hidden_decoder

        for i in range(len(n_hidden_decoder)):
            l_tmp_decoder = lasagne.layers.DenseLayer(l_prev_decoder,
                                                      num_units=n_hidden_decoder[-(i + 1)],
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=self.transf)

            l_prev_decoder = l_tmp_decoder

        l_in = lasagne.layers.InputLayer(shape=(batch_size, n_in))
        self.model = VAELayer(l_in,
                              encoder=l_prev_encoder,
                              decoder=l_prev_decoder,
                              latent_size=n_out,
                              x_distribution='gaussian',
                              qz_distribution='gaussian',
                              pz_distribution='gaussian')
        self.x = T.matrix('x')

    def build_model(self, train_x, update, update_args):
        self.train_x = train_x
        self.update = update
        self.update_args = update_args
        self.index = T.ivector('index')
        self.slice = self.index[:self.batch_size]
        # x = self.srng.binomial(size=self.x.shape, n=1, p=self.x)
        log_pz, log_qz_given_x, log_px_given_z = self.model.get_log_distributions(self.x)
        loss_eval = (log_pz + log_px_given_z - log_qz_given_x).sum()
        loss_eval /= self.batch_size

        all_params = get_all_params(self.model)
        updates = self.update(-loss_eval, all_params, *self.update_args)

        train_model = theano.function([self.index], loss_eval, updates=updates,
                                      givens={self.x: self.train_x[self.slice], },)

        return train_model

    def draw_sample(self, z):
        return self.model.draw_sample(z)

    def get_output(self, dat):
        z, _, _ = self.model.get_z_mu_sigma(dat)
        return z

    def get_reconstruction(self, z):
        return self.model.decoder_output(z)

    def get_params(self):
        return get_all_params(self.model)




# from here we apply our data
cor = loadmat('../data/come_done_COR.mat')['COR']
rec_cor = np.zeros_like(cor)
# so the dataset is created online, for every timestep I need to compute the C-mat and train a new VAE with that
time, freq, rates = cor.shape
d = freq * rates / 2
c_mat = np.zeros(shape=(freq+1, d))
n_epochs = 20
mask = np.array([1, 0])
all_params = None
c_mat_sym = T.dmatrix()

model = VAE(d, [300, 400], 2, trans_func=rectify, batch_size=128)
rng = np.random.RandomState(42)
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

    m_min = np.min(c_mat)
    m_max = np.max(c_mat)
    # print m_min
    # print m_max
    c_mat = (c_mat - m_min)
    # print c_mat
    # m_max = np.max(np.abs(c_mat))
    # print m_max
    # c_mat /= m_max
    x_prime = x_prime.transpose()
    c_mat[-1, :] = x_prime
    # c_mat is ready to be fed and train the VAE the whole training set has 'freq' #samples
    shared_c_mat = theano.shared(np.asarray(c_mat, dtype=theano.config.floatX), borrow=True)
    shared_x_prime = theano.shared(np.asarray(x_prime, dtype=theano.config.floatX), borrow=True)

    # print c_mat.shape

    train_model = model.build_model(shared_c_mat, adam, update_args=(1e-4,))
    eval_train = []
    eval_test = []
    eval_valid = []
    start_time = tm.time()
    for epoch in range(n_epochs):
        idx = np.random.permutation(freq+1).astype('int32')
        for i in range(4):
            eval_train += [train_model(idx)]
        log_pz, log_qz_given_x, log_px_given_z = model.model.get_log_distributions(shared_c_mat)
    end_time = tm.time() - start_time
    print "[time %i, time %.2f ,train %.10f]" % (t, end_time, eval_train[epoch])
    z = model.get_output(shared_x_prime)
    x_recon_mu, x_recon_sigma = model.get_reconstruction(z)
    x_recon_mu = x_recon_mu.reshape((d,))

    z_mat = model.get_output(shared_c_mat)
    c_recon_mu, c_recon_sigma = model.get_reconstruction(z_mat)

    # samples = np.zeros_like(c_mat)
    # for i in range(freq):
    #     samples[i, :] = rng.multivariate_normal(x_recon_mu.eval().reshape((d,)),
    #                                             np.diag(T.exp(x_recon_sigma).eval().reshape((d,))))
    # print samples.shape
    print x_recon_mu.eval().shape
    plt.figure
    plt.subplot(1, 2, 1)
    plt.imshow(c_mat, interpolation='nearest', aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(c_recon_mu.eval(), interpolation='nearest', aspect='auto')
    # tx = np.arange(d)
    # plt.plot(tx, c_recon_mu.eval()[0, :], 'b', tx, c_mat[0, :], 'r')
    plt.show()
    rec_cor[t, :, :rates/2] = x_recon_mu.eval().reshape(freq, rates/2)

savemat('come_done_filtered.mat', {'rec_cor': rec_cor})
