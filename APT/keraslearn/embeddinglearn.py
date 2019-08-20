# -*- utf-8 -*-
"""
Created on 15 Aug, 2019.

Word Embedding in keras.

@Author: Huang Zewen
"""
import sys
sys.path.append('../')

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard

from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# define class labels
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")

print(padded_docs)

model = Sequential()
model.add(Embedding(vocab_size, 12, input_length=max_length))
model.add(Flatten())
# model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(padded_docs, labels, epochs=50, verbose=0, callbacks=[tensorboard])

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))