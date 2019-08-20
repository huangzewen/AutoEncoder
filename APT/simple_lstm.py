# -*- utf-8 -*-
"""
Created on 14 Aug, 2019.

NN Module.

@Author: Huang Zewen
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from apt_reader import DATASET_PATH
from pandas import DataFrame, read_csv

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras import Sequential
from keras.layers import LSTM,  Dense, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
MAX_NB_WORDS = 5500
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100

dataset_df = read_csv(DATASET_PATH)
dataset_df = dataset_df.dropna()
dataset_df = dataset_df.sample(frac=1)

dataset_df.info()

print("Label Distributions: ", dataset_df['Label'].value_counts())

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(dataset_df['Log'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(dataset_df['Log'].values)
X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(dataset_df['Label']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

epochs = 5
batch_size = 1000


model = Sequential()
# The first layer is the embedded layer that uses 100 length vectors to represent each word.
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# SpatialDropout1D performs variational dropout in NLP models.
model.add(SpatialDropout1D(0.2))
# LSTM layer with 100 memory units.
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# The output layer must create 6 output values, one for each class.
model.add(Dense(6, activation='softmax'))
# Activation function is softmax for multi-class classification.
# Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001),tensorboard])


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
