from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat
from scipy.cluster.vq import whiten

train_x = ((np.array(loadmat('../data/normal_data.mat')['train_x'])))
train_y = np.log2((np.array(loadmat('../data/normal_data.mat')['train_y'])))
test_x = ((np.array(loadmat('../data/normal_data.mat')['test_x'])))
test_y = (np.array(loadmat('../data/normal_data.mat')['test_y']))
#train_x = (np.array(loadmat('../data/all_data.mat')['data_x']))
#train_y = (np.array(loadmat('../data/all_data.mat')['data_y']))
#validate = (np.array(loadmat('../data/validate.mat')['validate']))
#train_y /= max(train_y)
#test_y /= max(test_y)

m_train = np.mean(train_x, axis=0)
std_train = np.std(train_x, axis=0)

train_x = (train_x - m_train)/std_train
test_x = (test_x - m_train)/std_train

#train_x = whiten(np.concatenate((train_x), axis=1))
#test_x = whiten(np.concatenate((test_x), axis=1))
#validate = whiten(np.concatenate((np.sqrt(validate), np.log2(validate)), axis=1))

model = Sequential()
model.add(Dense(50, input_dim=len(train_x[0])))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(15))
model.add(Activation('linear'))
model.add(Dropout(0.5))
model.add(Dense(1))
optimizer = Adam()
model.compile(loss='mean_absolute_error', optimizer=optimizer)
model.fit(train_x, train_y, nb_epoch=2000, batch_size=32)
pred_val = model.predict(test_x, batch_size=32)
tot = np.sqrt(((2**(pred_val) - test_y)**2).mean())
print tot
#red = model.predict(validate)
#np.savetxt("foo3.csv", pred, delimiter=",")