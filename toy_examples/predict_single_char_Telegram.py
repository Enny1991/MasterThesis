from __future__ import absolute_import
from __future__ import print_function
# for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.layers.recurrent import LSTM


import numpy as np
import random
import sys

np.random.seed(42)  # for reproducibility

text = open('telegram_char.txt', 'r').read() # should be simple plain text file
chars = list(set(text))
data_size, vocab_size = len(text), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_indices = dict((ch, i) for i,ch in enumerate(chars))
indices_char = dict((i, ch) for i,ch in enumerate(chars))


maxlen = 30
step = 3

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen,step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('nb sequences: ', len(sentences))

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: 2 stacked LSTM

model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars))))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=4)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:  # different temperatures
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iterations in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()






