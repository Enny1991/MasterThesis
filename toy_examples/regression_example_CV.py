from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import csv
from keras.optimizers import Adam, Adagrad
import numpy as np
from scipy.io import loadmat, savemat

data_x = (np.array(loadmat('../data/all_data.mat')['data_x']))
data_y = (np.array(loadmat('../data/all_data.mat')['data_y']))
#validate = np.log2(np.array(loadmat('../data/validate.mat')['validate']))
#train_y /= max(train_y)
#test_y /= max(test_y)

#train_x = np.concatenate((train_x), axis=1)
#test_x = np.concatenate((test_x), axis=1)

#import pdb; pdb.set_trace()

np.random.seed(42)
perm = np.random.permutation(len(data_x))
samples_per_bucket = np.floor(len(data_x)/10)
tot = np.zeros(10)

for k in range(10):
    train_x = np.concatenate((data_x[0:k*samples_per_bucket], data_x[(k+1)*samples_per_bucket:]), axis=0)
    train_y = np.concatenate((data_y[0:k*samples_per_bucket], data_y[(k+1)*samples_per_bucket:]), axis=0)
    test_x = data_x[k*samples_per_bucket:(k+1)*samples_per_bucket]
    test_y = data_y[k*samples_per_bucket:(k+1)*samples_per_bucket]

    train_x = np.concatenate((np.sqrt(train_x), np.log2(train_x)), axis=1)
    test_x = np.concatenate((np.sqrt(test_x), np.log2(test_x)), axis=1)

    model = Sequential()
    model.add(Dense(1000, input_dim=len(train_x[0])))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Activation('linear'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.fit(train_x, train_y, nb_epoch=2000, batch_size=32)
    tot[k] = model.evaluate(test_x, test_y, batch_size=32)
print np.mean(tot)
print np.std(tot)

#pred = model.predict(validate)
#np.savetxt("foo.csv", pred, delimiter=",")