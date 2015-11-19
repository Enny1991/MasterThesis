from keras.models import Sequential
from keras.utils import np_utils
import lasagne
import theano
import theano.tensor as T
import os
from scipy.io import loadmat, savemat
import numpy as np
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
data_x = loadmat('../data/ml2_lowdim.mat')['low_dim']
data_y = loadmat('../data/ml2.mat')['train_y']
val_x = data_x[2025:]

#white



train_x = data_x[:2025]
train_y = data_y
#test_x = data_x[1800:]
#test_y = data_y[1800:]
mean_x = np.mean(train_x)
std_x = np.std(train_x)

train_x = (train_x - mean_x) / std_x
#test_x = (test_x - mean_x) / std_x
val_x = (val_x - mean_x) / std_x

# convert class vectors to binary class matrices
train_y = np_utils.to_categorical(train_y, 3)
#test_y = np_utils.to_categorical(test_y, 3)

print train_x.shape
#print test_x.shape
print val_x.shape

model = Sequential()
model.add(Dense(30, input_dim=(len(train_x[1]))))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Activation('tanh'))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('softmax'))
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(train_x, train_y, show_accuracy=True, nb_epoch=1000, batch_size=32)
tot = model.evaluate(train_x, train_y, show_accuracy=True)
print tot[1]
pred_val = model.predict(val_x)
pred_labels = np.argmax(pred_val, axis=-1)
savemat('prediction_nn_93175_with_prob.mat', {'pred_labels': pred_labels,'prob':pred_val})