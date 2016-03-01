from keras.models import Sequential
from keras.utils import np_utils
import lasagne
import theano
import theano.tensor as T
import os
from scipy.io import loadmat, savemat
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat
from scipy.cluster.vq import whiten
from sklearn.kernel_approximation import RBFSampler
from matplotlib import pyplot as plt

np.random.seed(42)
batch_size = 16
n_features = 7

#load
data_x = loadmat('../data/ml2.mat')['train_x']
data_y = loadmat('../data/ml2.mat')['train_y']
test_x = loadmat('../data/ml2.mat')['test_x']

boot = len(test_x)
#white
train_x = data_x
train_y = data_y
total_samples = len(train_x)
#train_x = np.concatenate((train_x, np.log10(train_x), np.sqrt(train_x)), axis=1)
#test_x = np.concatenate((test_x, np.log10(test_x), np.sqrt(test_x)), axis=1)
val_x = data_x[1900:]
val_y = data_y[1900:]
#val_x = np.concatenate((val_x, np.log10(val_x),np.sqrt(val_x)), axis=1)
mean_x = np.mean(train_x)
std_x = np.std(train_x)

train_x = (train_x - mean_x) / std_x
test_x = (test_x - mean_x) / std_x
val_x = (val_x - mean_x) / std_x
# convert class vectors to binary class matrices
train_y = np_utils.to_categorical(train_y, 3)
val_y = np_utils.to_categorical(val_y, 3)

print train_x.shape

print val_x.shape
print val_y.shape
print test_x.shape
print train_y.shape

n_boot = 1000
all_class = np.zeros((n_boot, boot))
all_class_validation = np.zeros((n_boot, len(val_x)))
# boostrap 1800

for i in range(n_boot):
    # idx = np.random.randint(0, total_samples, total_samples)
    # print idx.shape
    b_train_x = train_x
    b_train_y = train_y
    print b_train_x.shape
    model = Sequential()
    model.add(Dense(3, input_dim=(len(b_train_x[1]))))
    model.add(Activation('tanh'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(b_train_x, b_train_y, show_accuracy=True, nb_epoch=7, batch_size=128)
    pred_val = model.predict(test_x)
    pred_labels = np.argmax(pred_val, axis=-1)
    all_class[i, :] = pred_labels

#tot = model.evaluate(test_x, test_y, show_accuracy=True)
savemat('ensemble.mat', {'all_labels': all_class})


# savemat('prediction_nn_dropout.mat', {'pred_labels': pred_labels,'prob':pred_val})