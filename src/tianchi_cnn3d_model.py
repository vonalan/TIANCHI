from __future__ import print_function
import os
import sys

import numpy as np
import tensorflow as tf

# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
# import tianchi_data_processor_for_seperate_model_dev as tcdpsm


class cnn3d(object):
    def __init__(self, activation = tf.nn.relu):
        self.name = 'cnn3d'

        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101
        self.n_channel = 1

        # self.n_cnn_rnn = 1024
        # self.n_rnn_lgm = 1

        self.n_hidden = 1024
        self.n_input = self.n_steps * self.n_height * self.n_width * self.n_length * self.n_channel
        self.n_output = 1

        self.eta = 0.1
        self.dropout = 0.75
        self.activation = activation

        # network_weights = self._initialize_weights()
        # self.weights = self._initialize_weights()
        # self.biases = self._initialize_biases()
        # self.dropout =

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.t = tf.placeholder(tf.float32, [None, self.n_output])
        self.y = self._conv_net()

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.y, self.t), 2.0)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # Create some wrappers for simplicity
    def _conv_net(self):
        x = tf.reshape(self.x, [-1, 15, 4, 101, 101])
        x = tf.transpose(x,[0,1,3,4,2])[:,4:,:,:,3]
        x = tf.reshape(x,[-1,11,101,101,1])

        conv1 = tf.layers.conv3d(x, 64, [4, 8, 8],
                                strides = (2,4,4),
                                padding = 'same',
                                activation=tf.nn.relu,
                                name = 'conv1')
        pool1 = tf.layers.max_pooling3d(conv1, [4, 8, 8], [2,4,4],
                                    padding = 'same',
                                    name = 'pool1')


        conv2 = tf.layers.conv3d(pool1, 128, [2,4,4],
                                strides = (1,2,2),
                                padding="same",
                                activation=tf.nn.relu,
                                name = 'conv2')
        pool2 = tf.layers.max_pooling3d(conv2, [2,4,4], [1,2,2],
                                        padding = 'same',
                                        name = 'pool2')

        pool1_flat = tf.reshape(pool2, [-1, 3*2*2*128])

        # dense1 = tf.layers.dense(pool1_flat,
        #                         units=2048,
        #                         activation=tf.nn.relu,
        #                         name = 'dense1')
        # dropout = tf.layers.dropout(dense1,
        #                             rate=0.5,
        #                             name = 'dropout')
        # dense2 = tf.layers.dense(dropout,
        #                         units=2048,
        #                         activation=tf.nn.relu,
        #                         name = 'dense2')

        y = tf.layers.dense(pool1_flat,
                            units=1,
                            activation = tf.nn.relu,
                            name = 'y')

        return y 

    def partial_fit(self, X, T):
        rmse, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x:X, self.t:T})
        return rmse

    def predict(self, X, T):
        rmse, valid = self.sess.run((self.cost, self.y), feed_dict={self.x:X, self.t:T})
        return rmse, valid

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.y, feed_dict={self.x:X})

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


if __name__ == '__main__': 
    model = cnn3d()