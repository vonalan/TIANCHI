# coding=utf-8
# Author: kingdom

from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class rnn2d(object):
    # tf Graph input
    def __init__(self, activation = tf.nn.relu):
        self.name = 'rnn_2d'

        self.n_steps = 15
        self.n_bovfs = 128

        self.n_hidden = 128
        self.n_input = self.n_steps * self.n_bovfs
        self.n_output = 1

        self.eta = 1.0
        # self.dropout = 0.75
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        self.x = tf.placeholder("float", [None, self.n_input])
        self.t = tf.placeholder("float", [None, self.n_output])
        self.y = self._rec_net()

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.y, self.t), 2.0)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
        }

        return weights

    def _initialize_biases(self):
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }

        return biases

    def _rec_net(self):
        x = tf.reshape(self.x, [-1, self.n_steps, self.n_bovfs])
        x = tf.unstack(x, self.n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        out = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        return tf.nn.relu(out)

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