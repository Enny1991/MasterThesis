import theano
import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer, Layer
from lasagne.nonlinearities import linear, rectify, identity
from lasagne.objectives import squared_error
from lasagne.layers import get_all_layers, get_output, get_all_params, InputLayer
from lasagne.updates import adam
from matplotlib import pyplot as plt
from lasagne import init
from theano.tensor.shared_randomstreams import RandomStreams
import math


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


class BayesBackpropLayer(Layer):

    def __init__(self, incoming, num_units, W=init.Normal(0.05), b=init.Normal(0.05), nonlinearity=rectify,
                 prior_sd=T.exp(-3), **kwargs):
        super(BayesBackpropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams()

        self.num_units = num_units
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.nonlinearity = (identity if nonlinearity is None else nonlinearity)
        self.prior_sd = prior_sd

        self.W = T.zeros((self.num_inputs, num_units))
        self.W_mu = self.add_param(W, (self.num_inputs, num_units), name="W_mu")
        self.W_logsigma = self.add_param(W, (self.num_inputs, num_units), name="W_sigma")
        self.W_params = [self.W, self.W_mu, self.W_logsigma]
        self.b = T.zeros((num_units,))
        self.b_mu = self.add_param(b, (num_units,))
        self.b_logsigma = self.add_param(b, (num_units,))
        self.params = [self.W_mu, self.W_logsigma, self.b_mu, self.b_logsigma]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        if deterministic:
            activation = T.dot(input, self.W_mu) + self.b_mu.dimshuffle('x', 0)
        else:
            W = self.get_W()
            b = self.get_b()
            activation = T.dot(input, W) + b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    def get_y_mu_sigma(self, x):
        layers = get_all_layers(self)
        # output from sampled weights of all layers-1.
        z = get_output(layers[-2], x, deterministic=False)
        # sampled output of the final layer.
        y = self.nonlinearity(T.dot(z, self.get_W()) + self.get_b().dimshuffle('x', 0))
        # mean output of the final layer.
        y_mu = self.nonlinearity(T.dot(z, self.W_mu) + self.b_mu.dimshuffle('x', 0))
        # logsigma output of the final layer.
        y_logsigma = self.nonlinearity(T.dot(z, self.W_logsigma) + self.b_logsigma.dimshuffle('x', 0))
        return y, y_mu, y_logsigma

    def get_log_distributions(self, x, t, n_samples=1):
        #TODO: calculate the log distributions.
        def one_sample(_x, _t):
            y, y_mu, y_logsigma = self.get_y_mu_sigma(_x)
            # logP(D|w)
            _log_pd_given_w = normal2(_t, y, T.log(self.prior_sd ** 2)).sum()
            # logq(w) logp(w)
            _log_qw, _log_pw = 0., 0.
            layers = get_all_layers(self)[1:]
            for layer in layers:
                W = layer.W
                b = layer.b
                _log_qw += normal2(W, layer.W_mu, layer.W_logsigma * 2).sum()
                _log_qw += normal2(b, layer.b_mu, layer.b_logsigma * 2).sum()
                _log_pw += normal(W, 0., self.prior_sd).sum()
                _log_pw += normal(b, 0., self.prior_sd).sum()
            return _log_qw, _log_pw, _log_pd_given_w

        log_qw, log_pw, log_pd_given_w = 0., 0., 0.
        for i in range(n_samples):
            log_qw_temp, log_pw_tmp, log_pd_given_w_tmp = one_sample(x, t)
            log_qw += log_qw_temp
            log_pw += log_pw_tmp
            log_pd_given_w += log_pd_given_w_tmp

        log_qw /= n_samples
        log_pw /= n_samples
        log_pd_given_w /= n_samples
        return log_qw, log_pw, log_pd_given_w

    def get_params(self):
        return self.params

    def get_W(self):
        # TODO: Sample the weights and return.
        W = T.zeros(self.W_mu.shape)
        eps = self._srng.normal(size=self.W_mu.shape, avg=0., std=self.prior_sd)
        W += self.W_mu + T.log(1. + T.exp(self.W_logsigma)) * eps
        self.W = W
        return W

    def get_b(self):
        # TODO: Sample the bias and return.
        b = T.zeros(self.b_mu.shape)
        eps = self._srng.normal(size=self.b_mu.shape, avg=0., std=self.prior_sd)
        b += self.b_mu + T.log(1. + T.exp(self.b_logsigma)) * eps
        self.b = b
        return b


class VNN:
    def __init__(self, n_in, n_hid, n_out, trans_func=rectify, out_func=linear, W=init.Normal(0.05), b=init.Normal(0.05),
                batch_size=128, n_samples=10, prior_sd=T.exp(-3)):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.l_in = InputLayer((batch_size, n_in))
        self.batch_size = batch_size
        self.trans_func = trans_func
        self.out_func = out_func
        self.n_samples = n_samples

        # define model with lasagne
        l_prev = self.l_in
        for n_hid_l in self.n_hid:
            l_hidden = BayesBackpropLayer(l_prev, num_units=n_hid_l, W=W, b=b, nonlinearity=self.trans_func, prior_sd=prior_sd)
            l_prev = l_hidden

        self.model = BayesBackpropLayer(l_prev, num_units=self.n_out, nonlinearity=self.out_func)
        self.x = T.dmatrix('x')
        self.t = T.dmatrix('t')

    def build_model(self, train_x, train_y, test_x, test_y, update, update_args):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.update = update
        self.update_args = update_args
        self.index = T.iscalar('index')
        self.batch_slice = slice(self.index * self.batch_size, (self.index + 1) * self.batch_size)

        log_qw, log_pw, log_pd_given_w = self.model.get_log_distributions(self.x, self.t, self.n_samples)

        n_tot = self.train_x.shape[0].astype(theano.config.floatX)
        n_batches = n_tot / self.batch_size

        loss = ((1./n_batches) * (log_qw - log_pw) - log_pd_given_w).sum()/self.batch_size

        all_params = get_all_params(self.model)
        updates = self.update(loss, all_params, *self.update_args)

        train_model = theano.function(
            [self.index],
            loss,
            updates=updates,
            givens={
                self.x: self.train_x[self.batch_slice],
                self.t: self.train_y[self.batch_slice]
            },
        )

        test_model = theano.function(
            [self.index], loss,
            givens={
                self.x: self.test_x[self.batch_slice],
                self.t: self.test_y[self.batch_slice],
            },
        )
        return train_model, test_model

    def get_output(self, dat, deterministic=True):
        return lasagne.layers.get_output(self.model, dat, deterministic=deterministic)


n_epochs = 100
batch_size = 100
n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size
model = VNN(1, [50, 50], 1, n_samples=5, batch_size=batch_size)
eval_train = {}
eval_test = {}
train_model, test_model = model.build_model(train_x,
                                            train_y,
                                            test_x,
                                            test_y,
                                            adam, update_args=(0.001,))

for e in range(n_epochs):
    avg_cost = []
    for ind_batch in xrange(n_train_batches):
        mini_avg_cost = train_model(ind_batch)
        avg_cost.append(mini_avg_cost)
    eval_train[e] = np.mean(avg_cost)
    test_losses = [test_model(i) for i in xrange(n_test_batches)]

    eval_test[e] = np.mean(test_losses)

    print "[epoch,train,test];%i;%.10f;%.10f" % (e, eval_train[e], eval_test[e])

