"""
Created on 24 Jul, 2019

Module to test LSTM

@author: Huang Zewen
"""

import tensorflow as tf
import ptb_reader

DATA_PATH = "./data/simple-examples/data"

lstm_hidden_size = 10
batch_size = 100

num_steps = 50

train_data, valid_data, test_data, vocabulary = ptb_reader.ptb_raw_data(DATA_PATH)

print(train_data)
print(valid_data)
print(test_data)
print(vocabulary)