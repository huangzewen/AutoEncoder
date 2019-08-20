# -*- utf-8 -*-
"""
Created on 19 Aug, 2019.

Load saved apt_word2vec_model.

@Author: Huang Zewen
"""


import tensorflow as tf
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict


class WORD2VEC_UTIL(object):
    def __init__(self):
        self._sess = tf.Session()

        self._saver = tf.train.import_meta_graph('./apt_word2vec_model/apt_word2vec_weights_all.ckpt.meta')
        self._saver.restore(self._sess, tf.train.latest_checkpoint('./apt_word2vec_model'))
        self._vocab = self.load_vocab()

        self._int_2_char = dict((i, c) for i, c in enumerate(self._vocab))
        self._char_2_int = dict((c, i) for i, c in enumerate(self._vocab))

    def load_vocab(self):
        with open('apt_word2vec_model/vocab.txt', 'rb') as f:
            return pickle.load(f)

    def get_embeddings(self):
        embeddings = dict()
        for i in self._vocab:
            temp_a = np.zeros([1,len(self._vocab)])
            temp_a[0][self._char_2_int[i]] = 1
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name('input_x:0')
            temp_emb = self._sess.run(["predict_y:0"],feed_dict = {x:temp_a})
            temp_emb = np.array(temp_emb)
            #print(temp_emb.shape)
            embeddings[i] = temp_emb.reshape([len(self._vocab)])
        return embeddings

    def closest(self, embeddings, word, n):
        distances = dict()
        print(embeddings)
        for w in embeddings.keys():
            distances[w] = cosine_similarity(embeddings[w].reshape(1, -1),embeddings[word].reshape(1, -1))

        print(distances)
        d_sorted = OrderedDict(sorted(distances.items(),key = lambda x:x[1] ,reverse = True))
        s_words = list(d_sorted.keys())
        print(s_words)
        print(s_words[:n])

    def close(self):
        self._sess.close()


if __name__ == "__main__":
    apt_word_util = WORD2VEC_UTIL()
    embeddings = apt_word_util.get_embeddings()
    apt_word_util.closest(embeddings, 'issue', 10)
