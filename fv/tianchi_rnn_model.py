# coding=utf-8
# Author: kingdom

from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model as tcdpsm


class rnn2d(object):
    # tf Graph input
    def __init__(self, activation = tf.nn.relu):
        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101
        self.n_channel = 1

        # self.n_cnn_rnn = 1024
        # self.n_rnn_lgm = 1

        self.n_hidden = 128
        self.n_input = self.n_steps * self.n_height * self.n_width * self.n_length * self.n_channel
        self.n_output = 1

        self.eta = 1.0
        self.dropout = 0.75
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
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        x = tf.reshape(self.x, [-1, self.n_steps, self.n_height * self.n_width * self.n_length * self.n_channel])
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
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

    def saveModel(self, inmode, iter):
        if not os.path.exists('../data/rnn_model/'):
            os.mkdir('../data/rnn_model/')
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/rnn_model/rnn_model_inmode_{}_iter_{}.ckpt".format(inmode, iter))

    def loadModel(self, inmode, iter):
        saver = tf.train.Saver()
        saver.load(self.sess, "../data/rnn_model/rnn_model_inmode_{}_iter_{}.ckpt".format(inmode, iter))