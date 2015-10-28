from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat
from scipy.cluster.vq import whiten

# train_x = (np.array(loadmat('../data/normal_data.mat')['train_x']))
# train_y = (np.array(loadmat('../data/normal_data.mat')['train_y']))
# test_x = (np.array(loadmat('../data/normal_data.mat')['test_x']))
# test_y = (np.array(loadmat('../data/normal_data.mat')['test_y']))
train_x = (np.array(loadmat('../data/all_data.mat')['data_x']))
train_y = (np.array(loadmat('../data/all_data.mat')['data_y']))
validate = (np.array(loadmat('../data/validate.mat')['validate']))
#train_y /= max(train_y)
#test_y /= max(test_y)


train_x = whiten(np.concatenate((np.sqrt(train_x), np.log2(train_x)), axis=1))
#test_x = whiten(np.concatenate((np.sqrt(test_x), np.log2(test_x)), axis=1))
validate = whiten(np.concatenate((np.sqrt(validate), np.log2(validate)), axis=1))

model = Sequential()
model.add(Dense(50, input_dim=len(train_x[0])))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Activation('linear'))
model.add(Dropout(0.3))
model.add(Dense(1))
optimizer = Adam()
model.compile(loss='mean_absolute_error', optimizer=optimizer)
model.fit(train_x, train_y, nb_epoch=4000, batch_size=32)
#tot = model.evaluate(test_x, test_y, batch_size=32)
#print tot
pred = model.predict(validate)
np.savetxt("foo3.csv", pred, delimiter=",")