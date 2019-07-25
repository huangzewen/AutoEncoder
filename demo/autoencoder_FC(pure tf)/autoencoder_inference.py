# -*- coding: utf-8 -*-

"""
Created on 22 Jul, 2019.

Module for feed forward inference.

@author: Huang Zewen
"""

import tensorflow as tf

INPUT_NODE = 59
OUTPUT_NODE = 59

LAYER1_NODE = 32
LAYER2_NODE = 16
LAYER3_NODE = 16
LAYER4_NODE = 32


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope("layer3"):
        weights = get_weight_variable([LAYER2_NODE, LAYER3_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER3_NODE], initializer=tf.constant_initializer(0.0))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    with tf.variable_scope("layer4"):
        weights = get_weight_variable([LAYER3_NODE, LAYER4_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER4_NODE], initializer=tf.constant_initializer(0.0))
        layer4 = tf.nn.relu(tf.matmul(layer3, weights) + biases)

    with tf.variable_scope("layer5"):
        weights = get_weight_variable([LAYER4_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer5 = tf.matmul(layer4, weights) + biases

    return layer5
