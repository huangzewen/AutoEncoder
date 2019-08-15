# -*- utf-8 -*-
"""
Created on 14 Aug, 2019.

NN Module.

@Author: Huang Zewen
"""

import math
from sklearn.model_selection import train_test_split

from apt_reader import DATASET_PATH
from pandas import DataFrame, read_csv

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import  EarlyStopping

dataset_df = read_csv(DATASET_PATH)
dataset_df = dataset_df.dropna()

X = dataset_df.Log
Y = dataset_df.Label
le = LabelEncoder()

Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

#X_train, X_test is Series type; Y_train, Y_test is Array type
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

"""
Process the data
Tokenize the data and convert the text to sequences.
Add padding to ensure that all the sequences have the same shape
There are many ways of taking the max_len and here an arbitrary length of 150 is chosen.
"""

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
print(X_train)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words, 200, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


