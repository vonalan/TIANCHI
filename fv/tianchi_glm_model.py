'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import os 

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model as tcdpsm


class glm2d(object):
    def __init__(self, activation = tf.nn.relu):
        self.name = ''
        # self.mode = ''

        self.len_traj = 30720
        self.len_hof = 110592
        self.len_hog = 98304
        self.len_mbhx = 98304
        self.len_mbhy = 98304

        '''bug bug bug'''
        # self.n_input = self.len_traj + self.len_hof + self.len_hog + self.len_mbhx + self.len_mbhy
        self.n_input = self.len_traj
        # self.n_input = self.len_hof + self.len_hog
        # self.n_input = self.len_mbhx + self.len_mbhy
        # self.n_hidden = 1024
        self.n_output = 1

        self.eta = 0.1
        self.dropout = 0.4
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        self.x = tf.placeholder("float", [None, self.n_input])
        self.t = tf.placeholder("float", [None, self.n_output])
        self.y = self._out_net()

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.y, self.t), 2.0)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'f1': tf.Variable(tf.random_normal([self.n_input, self.n_output]))
            # 'o1': tf.Variable(tf.random_normal([self.n_hidden, self.n_output])),
            # 'f2': tf.Variable(tf.random_normal([self.n_steps, self.n_hidden])),
            # 'o2': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
        }
        return weights

    def _initialize_biases(self):
        biases = {
            'f1': tf.Variable(tf.random_normal([self.n_output])),
            # 'o1': tf.Variable(tf.random_normal([self.n_output])),
            # 'f2': tf.Variable(tf.random_normal([self.n_hidden])),
            # 'o2': tf.Variable(tf.random_normal([self.n_output]))
        }
        return biases

    def _out_net(self):
        # to overcome overfitting problem, dropout is adopted. 
        # x = tf.nn.dropout(self.x, self.dropout)

        # x = tf.reshape(self.x, [-1, self.n_steps * self.n_height * self.n_width * self.n_length])
        f1 = tf.add(tf.matmul(self.x, self.weights['f1']), self.biases['f1'])
        f1 = self.activation(f1)

        # o1 = tf.add(tf.matmul(f1, self.weights['o1']), self.biases['o1'])
        # o1 = self.activation(o1)

        # x = tf.reshape(o1, [-1, self.n_steps])
        # f2 = tf.add(tf.matmul(x, self.weights['f2']), self.biases['f2'])
        # f2 = self.activation(f2)
        # o2 = tf.add(tf.matmul(f1, self.weights['o2']), self.biases['o2'])
        # o2 = self.activation(o2)

        return f1


    def get_weights(self, key):
        return self.sess.run(self.weights[key])


    def partial_fit(self, X, T):
        rmse, opt = self.sess.run((self.rmse, self.optimizer), feed_dict={self.x:X, self.t:T})
        return rmse

    def predict(self, X, T):
        rmse, valid = self.sess.run((self.rmse, self.y), feed_dict={self.x:X, self.t:T})
        return rmse, valid

    def saveModel(self, i, iter):
        if not os.path.exists('../data/%s/'%self.name):
            os.mkdir('../data/%s/'%self.name)
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/%s/%s_%d.ckpt"%(self.name, self.name, iter))

    def loadModel(self, i, iter):
        saver = tf.train.Saver()
        saver.restore(self.sess, "../data/%s/%s_%d.ckpt"%(self.name, self.name, iter))

    def saveResults(self, prefix, iter, results):
        np.savetxt("../data/%s/%s_%s_%d.txt" % (self.name, prefix, self.name, iter), results)

    def loadResults(self, prefix, iter, results): 
        pass 