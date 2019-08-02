# -*- coding: utf-8 -*-
"""
Created on 28 Jul, 2019.

LSTM module to train, valid, test PTB data.

@author: Huang Zewen
"""


import numpy as np
import tensorflow as tf

import ptb_reader

DATA_PATH = "./data/simple-examples/data"

train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data(DATA_PATH)

HIDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 1000

LEARNING_RATE = 1.0
TRAINING_BATCH_SIZE = 20
TRAINING_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5  # Parameter to control the gradient

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDEN_SIZE)

        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                     output_kep_prob=KEEP_PROB)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)

        self.inital_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training: inputs = tf.nn.dropout(inputs, KEEP_PROB)

        outputs = []

        state = self.inital_state

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                cell_output, state = cell(inputs[:, time_step, :], state)

                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, HIDEN_SIZE])

        weight = tf.get_variable("weight", [HIDEN_SIZE, VOCAB_SIZE])

        bias = tf.get_variable("bias", [VOCAB_SIZE])

        logits = tf.matmul(output, weight) + bias


        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])],
                                                                  [tf.ones([batch_size*num_steps], dtype=tf.float32)])


        self.cost = tf.reduce_sum(loss)/batch_size

        self.final_state = state

        if not is_training: return

        trainable_variables = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))





