# -*- coding: utf-8 -*-
"""
Created on 23 Jul, 2019.

Module to train the Full-Connection AutoEncoder.

@author: Huang Zewen
"""


import os

import tensorflow as tf
import autoencoder_inference
import dataminer

import numpy as np

BATCH_SIZE = 500
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
EPOCHS = 200
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

df_train = dataminer.get_train_0_x_rescaled()

def train():
    x = tf.placeholder(tf.float32, [None, autoencoder_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, autoencoder_inference.OUTPUT_NODE], name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = autoencoder_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    mse = tf.losses.mean_squared_error(y_, y)

    loss = mse + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               int(len(df_train)/BATCH_SIZE),
                                               LEARNING_RATE_DECAY)

    adam = tf.train.AdadeltaOptimizer(learning_rate)

    train_step = adam.minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for epoch in range(EPOCHS):
            x_batches = np.array_split(df_train, int(len(df_train)/BATCH_SIZE))
            y_batches = x_batches.copy()
            for i in range(int(len(df_train)/BATCH_SIZE)):
                batch_x = x_batches[i]
                batch_y = y_batches[i]
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: batch_x, y_: batch_y})

                # if i == int(len(df_train)/BATCH_SIZE) -1:
            print("After %d training epoch(s), loss on training batch is %g." %(epoch, loss_value))
            print("global step is: %d." % sess.run(global_step))

            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()