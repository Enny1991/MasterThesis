from keras.models import Sequential
from keras.layers.core import Dense, Activation



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat
from scipy.cluster.vq import whiten
from sklearn.kernel_approximation import RBFSampler

train_x = np.log2((np.array(loadmat('../data/normal_data.mat')['train_x'])))
train_y = np.log2((np.array(loadmat('../data/normal_data.mat')['train_y'])))
val_x = np.log2((np.array(loadmat('../data/normal_data.mat')['test_x'])))
val_y = (np.array(loadmat('../data/normal_data.mat')['test_y']))
#train_x = (np.array(loadmat('../data/all_data.mat')['data_x']))
#train_y = (np.array(loadmat('../data/all_data.mat')['data_y']))
test_x = np.log2(np.array(loadmat('../data/validate.mat')['validate']))
#train_y /= max(train_y)
#test_y /= max(test_y)
#rbf_feature = RBFSampler(gamma=0.001, random_state=2)
#train_x = rbf_feature.fit_transform(train_x)
#data_y = rbf_feature.fit_transform(data_x)
#val_x = rbf_feature.fit_transform(val_x)
#test_x = rbf_feature.fit_transform(test_x)

#
a1 = (train_x[:, 0] * train_x[:, 13]).reshape(len(train_x), 1)
b1 = (test_x[:, 0] * test_x[:, 13]).reshape(len(test_x), 1)
a2 = (train_x[:, 0] * train_x[:, 3]).reshape(len(train_x), 1)
b2 = (test_x[:, 0] * test_x[:, 3]).reshape(len(test_x), 1)
a3 = (train_x[:, 0] * train_x[:, 5]).reshape(len(train_x), 1)
b3 = (test_x[:, 0] * test_x[:, 5]).reshape(len(test_x), 1)
a5 = (train_x[:, 0] * train_x[:, 0]).reshape(len(train_x), 1)
b5 = (test_x[:, 0] * test_x[:, 0]).reshape(len(test_x), 1)
a4 = ((train_x[:, 0] * train_x[:, 13]* train_x[:, 5])**3).reshape(len(train_x), 1)
b4 = ((test_x[:, 0] * test_x[:, 13]* test_x[:, 5])**3).reshape(len(test_x), 1)
a6 = (train_x[:, 3] * train_x[:, 13]).reshape(len(train_x), 1)
b6 = (test_x[:, 3] * test_x[:, 13]).reshape(len(test_x), 1)
a7 = (train_x[:, 5] * train_x[:, 13]).reshape(len(train_x), 1)
b7 = (test_x[:, 5] * test_x[:, 13]).reshape(len(test_x), 1)
a8 = ((train_x[:, 5] * train_x[:, 13])**3).reshape(len(train_x), 1)
b8 = ((test_x[:, 5] * test_x[:, 13])**3).reshape(len(test_x), 1)
b9 = (test_x[:, 0] * test_x[:, 1]).reshape(len(test_x), 1)
b10 = (test_x[:, 0] * test_x[:, 2]).reshape(len(test_x), 1)
b11 = (test_x[:, 0] * test_x[:, 4]).reshape(len(test_x), 1)
b12 = (test_x[:, 0] * test_x[:, 6]).reshape(len(test_x), 1)
b13 = (test_x[:, 0] * test_x[:, 7]).reshape(len(test_x), 1)
b14 = (test_x[:, 0] * test_x[:, 8]).reshape(len(test_x), 1)
b15 = (test_x[:, 0] * test_x[:, 9]).reshape(len(test_x), 1)
b16 = (test_x[:, 0] * test_x[:, 10]).reshape(len(test_x), 1)
a9 = (train_x[:, 0] * train_x[:, 1]).reshape(len(train_x), 1)
a10 = (train_x[:, 0] * train_x[:, 2]).reshape(len(train_x), 1)
a11 = (train_x[:, 0] * train_x[:, 4]).reshape(len(train_x), 1)
a12 = (train_x[:, 0] * train_x[:, 6]).reshape(len(train_x), 1)
a13 = (train_x[:, 0] * train_x[:, 7]).reshape(len(train_x), 1)
a14 = (train_x[:, 0] * train_x[:, 8]).reshape(len(train_x), 1)
a15 = (train_x[:, 0] * train_x[:, 9]).reshape(len(train_x), 1)
a16 = (train_x[:, 0] * train_x[:, 10]).reshape(len(train_x), 1)
#
#
c1 = (val_x[:, 0] * val_x[:, 13]).reshape(len(val_x), 1)
#
c2 = (val_x[:, 0] * val_x[:, 3]).reshape(len(val_x), 1)
#
c3 = (val_x[:, 0] * val_x[:, 5]).reshape(len(val_x), 1)
#
c5 = (val_x[:, 0] * val_x[:, 0]).reshape(len(val_x), 1)
#
c4 = ((val_x[:, 0] * val_x[:, 13]* val_x[:, 5])**3).reshape(len(val_x), 1)
#
c6 = (val_x[:, 3] * val_x[:, 13]).reshape(len(val_x), 1)
#
c7 = (val_x[:, 5] * val_x[:, 13]).reshape(len(val_x), 1)
c8 = ((val_x[:, 5] * val_x[:, 13])**3).reshape(len(val_x), 1)
c9 = (val_x[:, 0] * val_x[:, 1]).reshape(len(val_x), 1)
c10 = (val_x[:, 0] * val_x[:, 2]).reshape(len(val_x), 1)
c11 = (val_x[:, 0] * val_x[:, 4]).reshape(len(val_x), 1)
c12 = (val_x[:, 0] * val_x[:, 6]).reshape(len(val_x), 1)
c13 = (val_x[:, 0] * val_x[:, 7]).reshape(len(val_x), 1)
c14 = (val_x[:, 0] * val_x[:, 8]).reshape(len(val_x), 1)
c15 = (val_x[:, 0] * val_x[:, 9]).reshape(len(val_x), 1)
c16 = (val_x[:, 0] * val_x[:, 10]).reshape(len(val_x), 1)
#
train_x = (np.concatenate((np.power(train_x,1./3),1./(train_x[:, 0].reshape(len(train_x), 1)),
                            1./(train_x[:, 3].reshape(len(train_x), 1)),
                            1./(train_x[:, 5].reshape(len(train_x), 1)),
                            1./(train_x[:, 13].reshape(len(train_x), 1)),
                            np.sqrt(train_x[:, 0].reshape(len(train_x), 1)),
                            np.sqrt(train_x[:, 3].reshape(len(train_x), 1)),
                            np.sqrt(train_x[:, 5].reshape(len(train_x), 1)),
                            np.sqrt(train_x[:, 13].reshape(len(train_x), 1)),
                            np.log2(train_x[:, 0].reshape(len(train_x), 1)),
                            np.log2(train_x[:, 3].reshape(len(train_x), 1)),
                            np.log2(train_x[:, 5].reshape(len(train_x), 1)),
                            np.log2(train_x[:, 13].reshape(len(train_x), 1)),
                            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16,
                            np.sqrt(a1),
                            np.sqrt(a2),
                            np.sqrt(a3),
                            np.sqrt(a4),
                            np.sqrt(a5),
                            np.sqrt(a6),
                            np.sqrt(a7)), axis=1))
test_x = (np.concatenate((np.power(test_x,1./3),1./(test_x[:, 0].reshape(len(test_x), 1)),
                           1./(test_x[:, 3].reshape(len(test_x), 1)),
                           1./(test_x[:, 5].reshape(len(test_x), 1)),
                           1./(test_x[:, 13].reshape(len(test_x), 1)),
                           np.sqrt(test_x[:, 0].reshape(len(test_x), 1)),
                           np.sqrt(test_x[:, 3].reshape(len(test_x), 1)),
                           np.sqrt(test_x[:, 5].reshape(len(test_x), 1)),
                           np.sqrt(test_x[:, 13].reshape(len(test_x), 1)),
                           np.log2(test_x[:, 0].reshape(len(test_x), 1)),
                           np.log2(test_x[:, 3].reshape(len(test_x), 1)),
                           np.log2(test_x[:, 5].reshape(len(test_x), 1)),
                           np.log2(test_x[:, 13].reshape(len(test_x), 1)),
                           b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16,
                           np.sqrt(b1),
                           np.sqrt(b2),
                           np.sqrt(b3),
                           np.sqrt(b4),
                           np.sqrt(b5),
                           np.sqrt(b6),
                           np.sqrt(b7)), axis=1))

val_x = (np.concatenate((np.power(val_x, 1./3),1./(val_x[:, 0].reshape(len(val_x), 1)),
                          1./(val_x[:, 3].reshape(len(val_x), 1)),
                          1./(val_x[:, 5].reshape(len(val_x), 1)),
                          1./(val_x[:, 13].reshape(len(val_x), 1)),
                          np.sqrt(val_x[:, 0].reshape(len(val_x), 1)),
                          np.sqrt(val_x[:, 3].reshape(len(val_x), 1)),
                          np.sqrt(val_x[:, 5].reshape(len(val_x), 1)),
                          np.sqrt(val_x[:, 13].reshape(len(val_x), 1)),
                          np.log2(val_x[:, 0].reshape(len(val_x), 1)),
                          np.log2(val_x[:, 3].reshape(len(val_x), 1)),
                          np.log2(val_x[:, 5].reshape(len(val_x), 1)),
                          np.log2(val_x[:, 13].reshape(len(val_x), 1)),
                          c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16,
                          np.sqrt(c1),
                          np.sqrt(c2),
                          np.sqrt(c3),
                          np.sqrt(c4),
                          np.sqrt(c5),
                          np.sqrt(c6),
                          np.sqrt(c7)), axis=1))
m_train = np.mean(train_x, axis=0)
m_test = np.mean(test_x, axis=0)
std_train = np.std(train_x, axis=0)
std_test = np.std(test_x, axis=0)
m_val = np.mean(val_x, axis=0)
std_val = np.std(val_x, axis=0)
train_x = (train_x - m_train)/std_train
test_x = (test_x - m_train)/std_train
val_x = (val_x - m_train)/std_train


# AE
model = Sequential()
model.add(Dense(15, input_dim=len(train_x[0])))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(train_x[0])))
optimizer = Adam()
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.fit(train_x, train_x, nb_epoch=2000, batch_size=64)
score = model.evaluate(test_x, test_x, batch_size=16)
train_x = model.predict(train_x)
test_x = model.predict(test_x)
val_x = model.predict(val_x)
#

m_train = np.mean(train_x, axis=0)
m_test = np.mean(test_x, axis=0)
std_train = np.std(train_x, axis=0)
std_test = np.std(test_x, axis=0)
m_val = np.mean(val_x, axis=0)
std_val = np.std(val_x, axis=0)
train_x = (train_x - m_train)/std_train
test_x = (test_x - m_train)/std_train
val_x = (val_x - m_train)/std_train


model = Sequential()
model.add(Dense(len(train_x[0]), input_dim=len(train_x[0])))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
optimizer = Adam()
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(train_x, train_y, validation_data=(val_x, val_y), show_accuracy=True, nb_epoch=4000, batch_size=64)
#tot = model.evaluate(val_x, val_y, batch_size=128)
pred_val = model.predict(val_x)
l = 2**(pred_val)
print l.shape
print val_y.shape
tot = np.sqrt(((2**(pred_val) - val_y)**2).mean())
print tot
pred = model.predict(test_x)
np.savetxt("foo_true.csv", 2**pred, delimiter=",")





