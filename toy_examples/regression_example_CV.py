from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat
from scipy.cluster.vq import whiten

data_x = (np.array(loadmat('../data/pure_data.mat')['data_x']))
data_y = (np.array(loadmat('../data/pure_data.mat')['data_y']))
val_x = (np.array(loadmat('../data/pure_data.mat')['validate']))
#train_y /= max(train_y)
#test_y /= max(test_y)

#train_x = np.concatenate((train_x), axis=1)
#test_x = np.concatenate((test_x), axis=1)

#import pdb; pdb.set_trace()

np.random.seed(43)
perm = np.random.permutation(len(data_x))
samples_per_bucket = np.floor(len(data_x)/10)
tot = np.zeros(10)

c1 = (val_x[:, 0] * val_x[:, 13]).reshape(len(val_x), 1)

c2 = (val_x[:, 0] * val_x[:, 3]).reshape(len(val_x), 1)

c3 = (val_x[:, 0] * val_x[:, 5]).reshape(len(val_x), 1)

c5 = (val_x[:, 0] * val_x[:, 0]).reshape(len(val_x), 1)

c4 = ((val_x[:, 0] * val_x[:, 13]* val_x[:, 5])**3).reshape(len(val_x), 1)

c6 = (val_x[:, 3] * val_x[:, 13]).reshape(len(val_x), 1)

c7 = (val_x[:, 5] * val_x[:, 13]).reshape(len(val_x), 1)
c8 = ((val_x[:, 5] * val_x[:, 13])**3).reshape(len(val_x), 1)

val_x = (np.concatenate((1./(val_x[:, 0].reshape(len(val_x), 1)),
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
                          c1, c2, c3, c4, c5, c6, c7, c8,
                          np.sqrt(c1),
                          np.sqrt(c2),
                          np.sqrt(c3),
                          np.sqrt(c4),
                          np.sqrt(c5),
                          np.sqrt(c6),
                          np.sqrt(c7)), axis=1))

for k in range(10):
    train_x = np.concatenate((data_x[0:k*samples_per_bucket], data_x[(k+1)*samples_per_bucket:]), axis=0)
    train_y = np.concatenate((data_y[0:k*samples_per_bucket], data_y[(k+1)*samples_per_bucket:]), axis=0)
    test_x = data_x[k*samples_per_bucket:(k+1)*samples_per_bucket]
    test_y = data_y[k*samples_per_bucket:(k+1)*samples_per_bucket]

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

    train_x = (np.concatenate((1./(train_x[:, 0].reshape(len(train_x), 1)),
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
                            a1, a2, a3, a4, a5, a6, a7, a8,
                            np.sqrt(a1),
                            np.sqrt(a2),
                            np.sqrt(a3),
                            np.sqrt(a4),
                            np.sqrt(a5),
                            np.sqrt(a6),
                            np.sqrt(a7)), axis=1))
    test_x = (np.concatenate((1./(test_x[:, 0].reshape(len(test_x), 1)),
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
                           b1, b2, b3, b4, b5, b6, b7, b8,
                           np.sqrt(b1),
                           np.sqrt(b2),
                           np.sqrt(b3),
                           np.sqrt(b4),
                           np.sqrt(b5),
                           np.sqrt(b6),
                           np.sqrt(b7)), axis=1))
    m_train = np.mean(train_x, axis=0)
    m_test = np.mean(test_x, axis=0)
    std_train = np.std(train_x, axis=0)
    std_test = np.std(test_x, axis=0)

    train_x = (train_x - m_train)/std_train
    test_x = (test_x - m_train)/std_train
    val_x = (val_x - m_train)/std_train

    model = Sequential()
    model.add(Dense(27, input_dim=27))
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(27*3))
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.fit(train_x, train_y, nb_epoch=1000, batch_size=64)
    tot[k] = model.evaluate(test_x, test_y, batch_size=64)
    pred = model.predict(val_x)
    np.savetxt("foo{}.csv".format(k), pred, delimiter=",")
print np.mean(tot)
print np.std(tot)
print(max(tot))

#pred = model.predict(val_x)
#np.savetxt("foo7.csv", pred, delimiter=",")