import theano
import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer
from lasagne.nonlinearities import linear, rectify
from lasagne.objectives import squared_error
from lasagne.updates import adam
from matplotlib import pyplot as plt


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

print test_x_unshared
print test_y_unshared


class MLP:
    def __init__(self, n_in, n_hid, n_out, trans_func=rectify, out_func=linear, batch_size=128, p_dropout=0.0):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.l_in = InputLayer((batch_size, n_in))
        self.batch_size = batch_size
        self.trans_func = trans_func
        self.out_func = out_func
        self.p_dropout = p_dropout

        # define model with lasgne
        l_prev = self.l_in
        for n_hid_l in self.n_hid:
            l_hidden = DenseLayer(l_prev, num_units=n_hid_l, nonlinearity=self.trans_func)
            if not self.p_dropout == 0.0:
                l_hidden = DropoutLayer(l_prev, p=self.p_dropout)
            l_prev = l_hidden

        self.model = DenseLayer(l_prev, num_units=self.n_out, nonlinearity=self.out_func)

        self.x = T.dmatrix('x')
        self.t = T.dmatrix('t')
        self.y = lasagne.layers.get_output(self.model, self.x, deterministic=True)

    def build_model(self, train_x, train_y, test_x, test_y, loss, update, update_args):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.update = update
        self.update_args = update_args
        self.index = T.iscalar('index')
        self.batch_slice = slice(self.index * self.batch_size, (self.index + 1) * self.batch_size)

        loss_train = loss(self.get_output(self.x, False), self.t).sum()/self.batch_size
        loss_eval = loss(self.get_output(self.x), self.t).sum()/self.batch_size

        all_params = lasagne.layers.get_all_params(self.model)
        updates = self.update(loss_train, all_params, *self.update_args)

        train_model = theano.function(
            [self.index],
            loss_train,
            updates=updates,
            givens={
                self.x: self.train_x[self.batch_slice],
                self.t: self.train_y[self.batch_slice]
            },
        )

        test_model = theano.function(
            [self.index], loss_eval,
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
model = MLP(1, [50, 50], 1, p_dropout=0.0, batch_size=batch_size)
eval_train = {}
eval_test = {}
train_model, test_model = model.build_model(train_x,
                                            train_y,
                                            test_x,
                                            test_y,
                                            squared_error,
                                            adam, update_args=(0.01,))

for e in range(n_epochs):
    avg_cost = []
    for ind_batch in xrange(n_train_batches):
        mini_avg_cost = train_model(ind_batch)
        avg_cost.append(mini_avg_cost)
    eval_train[e] = np.mean(avg_cost)
    test_losses = [test_model(i) for i in xrange(n_test_batches)]

    eval_test[e] = np.mean(test_losses)

    print "[epoch,train,test];%i;%.10f;%.10f" % (e, eval_train[e], eval_test[e])

fig = plt.figure()
plt.plot(eval_train.keys(), eval_train.values(), label='training data')
plt.plot(eval_test.keys(), eval_test.values(), label='testing data')
plt.legend()
plt.ylabel("Loss")
plt.xlabel('Epochs')
plt.show()

y = model.get_output(test_x, deterministic=False).eval()
fig = plt.figure()
plt.scatter(np.array(test_x_unshared), np.array(test_y_unshared), label="t", color=(1.0,0,0,0.2))
plt.scatter(np.array(test_x_unshared), np.array(y), label="y", color=(0,0.7,0,0.1))
plt.legend()
plt.show()
