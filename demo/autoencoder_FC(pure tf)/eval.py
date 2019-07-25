# -*- coding: utf-8 -*-
"""
Created on 24 Jul, 2019.

Module to evaluate full connectiong autoencoder.

@author: Huang Zewen
"""
import tensorflow as tf
import autoencoder_inference
import dataminer
import train
import numpy as np

from matplotlib import pyplot as plt

valid_0_x_rescaled = dataminer.get_valid_0_x_rescaled()
valid_1_x_rescaled = dataminer.get_valid_1_x_rescaled()


def evaluate(x_feed, y_feed, threshold):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, autoencoder_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, autoencoder_inference.OUTPUT_NODE], name='y-output')

        validate_feed = {
            x: x_feed,
            y_: y_feed
        }

        y = autoencoder_inference.inference(x, None)

        diff = tf.reduce_sum(tf.square(y - y_), 1)

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)

        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                # pred_y = sess.run(y, feed_dict=validate_feed)
                diff_y = sess.run(diff, feed_dict=validate_feed)

                # print("global_step: %s" % global_step)
                return diff_y

            else:
                print('No checkpoint file found.')
                return


def roc_curve():
    threshold = np.arange(1, 600, 1)

    x_feed_0 = valid_0_x_rescaled
    x_feed_1 = valid_1_x_rescaled

    precision_rate = []
    recall_rate = []
    fpr_rate = []

    for th in threshold:
        x_feed_0_diff = evaluate(x_feed_0, x_feed_0, th)
        x_feed_1_diff = evaluate(x_feed_1, x_feed_1, th)
        '''
        TP: Predict is Positive && Positive in fact
        FP: Predict is Positive && Negative in fact
        FN: Predict is Negative && Positibe in fact 
        Precision = TP/(TP+FP)
        Sensitivity = Recall = TPR = TP/(TP+FN)
        '''
        tp = sum(_>th for _ in x_feed_1_diff)
        fn = sum(_<=th for _ in x_feed_1_diff)
        fp = sum(_>th for _ in x_feed_0_diff)
        tn = sum(_<th for _ in x_feed_0_diff)
        precision_rate.append(tp/(tp+fp))
        recall_rate.append(tp/(tp+fn))

        fpr_rate.append(fp/(fp+tn))

    plt.plot(threshold, recall_rate, label="Recall", linewidth=5)
    plt.plot(threshold, precision_rate, label="Precision", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()


    plt.plot(fpr_rate, recall_rate, linewidth=5)
    plt.plot([0,1],[0,1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

roc_curve()



