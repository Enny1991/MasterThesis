import numpy as np

class LinearLayer():
    def __init__(self, num_inputs, num_units, scale=0.01):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.W = np.random.random((num_inputs, num_units)) * scale
        self.b = np.zeros(num_units)

    def __str__(self):
        return "LinearLayer(%i, %i)" % (self.num_inputs, self.num_units)

    def fprop(self, x, *args):
        self.x = x
        self.z = np.dot(self.x, self.W) + self.b
        return self.z

    def bprop(self, delta_in):
        x_t = np.transpose(self.x)
        self.grad_W = np.dot(x_t, delta_in)
        self.grad_b = delta_in.sum(axis=0)
        W_T = np.transpose(self.W)
        self.delta_out = np.dot(delta_in, W_T)
        return self.delta_out

    def update_params(self, lr):
        self.W = self.W - self.grad_W*lr
        self.b = self.b - self.grad_b*lr

    def linear(x):
        return x

    def relu(x):
        return max(0, x)