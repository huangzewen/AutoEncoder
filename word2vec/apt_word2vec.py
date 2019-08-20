# -*- utf-8 -*-
"""
Created on 18 Aug, 2019.

This module is used to run word2vec for apt by tensorflow.

@Author: Huang Zewen
"""


import numpy as np
from pandas import read_csv
import tensorflow as tf
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle


APT_DATASET_PATH = '../APT/data/dataset.csv'
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''


df = read_csv(APT_DATASET_PATH)
df = df.dropna()
df = df.sample(frac=1)

sentences = []
for log in df.Log:
    log = log.lower()
    for c in FILTERS:
        log = log.replace(c, ' ')
    sentences.append(log.split())


# build vocab
vocab = set()
for sentence in sentences:
    vocab |= set([word for word in sentence])
    # vocab |= set([word for word in sentence
    #               if len(word) > 1 and word.isalpha()])

char_2_int = dict((c, i) for i, c in enumerate(vocab))
int_2_char = dict((i, c) for i, c in enumerate(vocab))

X = []
Y = []

window_size = 2


def _proc_sentence(sentence):
    temp_dict = []
    for i in range(len(sentence)):
        a = i - window_size
        b = i + window_size
        cur_word = sentence[i]
        for z in range(a, i):
            if z >= 0:
                temp_dict.append([cur_word, sentence[z]])

        for z in range(i+1, b+1):
            if z < len(sentence):
                temp_dict.append([cur_word, sentence[z]])
        return temp_dict


training_samples = []


def proc_apt_text():
    for sentence in sentences:
        training_samples.extend(_proc_sentence(sentence))


proc_apt_text()


for pair in training_samples:
    tempx = np.zeros(len(vocab))
    tempy = np.zeros(len(vocab))
    tempx[char_2_int[pair[0]]] = 1
    tempy[char_2_int[pair[1]]] = 1
    X.append(tempx)
    Y.append(tempy)

embedding_size = 200
batch_size = 1000
epochs = 32

n_batches = int(len(X)/batch_size)
learning_rate = 0.001
x = tf.placeholder(tf.float32,shape=(None,len(vocab)),name='input_x')
# x_name = tf.identity(x, 'input_x')

y = tf.placeholder(tf.float32,shape=(None,len(vocab)))

w1 = tf.Variable(tf.random_normal([len(vocab),embedding_size]),dtype = tf.float32)
b1 = tf.Variable(tf.random_normal([embedding_size]),dtype = tf.float32)
w2 = tf.Variable(tf.random_normal([embedding_size,len(vocab)]),dtype = tf.float32)
b2 = tf.Variable(tf.random_normal([len(vocab)]),dtype = tf.float32)
hidden_y = tf.matmul(x,w1) + b1
_y = tf.matmul(hidden_y,w2) + b2

_y_name = tf.identity(_y, 'predict_y')

cost = tf.reduce_mean(tf.losses.mean_squared_error(_y,y))
tf.summary.scalar("loss", cost)
merged_summary_op = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print(y.shape)
print(_y.shape)
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
sess = tf.Session()
sess.run(init)

summary_writer=tf.summary.FileWriter('./apt_word2vec_log',graph=tf.get_default_graph())
for epoch in range(5):
    avg_cost = 0
    for i in range(n_batches-1):
        batch_x = X[i*batch_size:(i+1)*batch_size]
        batch_y = Y[i*batch_size:(i+1)*batch_size]
        #print(batch_x.shape)
        _,c = sess.run([optimizer,cost],feed_dict = {x: batch_x,y: batch_y})
        #print(test.shape)
        summary=sess.run(merged_summary_op,feed_dict = {x: batch_x,y: batch_y})
        avg_cost += c/n_batches
    print('Epoch',epoch,' - ',avg_cost)
save_path = saver.save(sess,'apt_word2vec_model/apt_word2vec_weights_all.ckpt')


def save_vocab():
    with open('apt_word2vec_model/vocab.txt', 'wb') as f:
        pickle.dump(vocab, f)

save_vocab()
embeddings = dict()
for i in vocab:
    temp_a = np.zeros([1,len(vocab)])
    temp_a[0][char_2_int[i]] = 1
    # temp_emb = sess.run([_y],feed_dict = {x:temp_a})
    temp_emb = sess.run(["predict_y:0"],feed_dict = {x: temp_a})
    temp_emb = np.array(temp_emb)
    #print(temp_emb.shape)
    embeddings[i] = temp_emb.reshape([len(vocab)])
    #print(embeddings[i].shape)


def closest(word,n):
    distances = dict()
    for w in embeddings.keys():
        distances[w] = cosine_similarity(embeddings[w],embeddings[word])
    d_sorted = OrderedDict(sorted(distances.items(),key = lambda x:x[1] ,reverse = True))
    s_words = d_sorted.keys()
    print(s_words[:n])


labels = []
tokens = []
for w in embeddings.keys():
    labels.append(w)
    tokens.append(embeddings[w])
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))

for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()


sess.close()
