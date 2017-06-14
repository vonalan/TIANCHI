'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model_dev as tcdpsm


class cnn2d(object):
    def __init__(self, n_input, n_hidden, n_output, n_steps, dropout, activation = tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_steps = n_steps
        self.dropout = dropout
        self.activation = activation

        # network_weights = self._initialize_weights()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        # self.dropout =

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def _initialize_weights(self):
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            ''''''
            'wc1': tf.Variable(tf.random_normal([5, 5, 4, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([26 * 26 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_output]))
        }

        return weights

    def _initialize_biases(self):
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }

        return biases


class rnn2d(object):
    # tf Graph input
    def __init__(self, n_input, n_hidden, n_output, n_steps, dropout, activation = tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_steps = n_steps
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

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


class cnnxrnn(object):
    def __init__(self, n_input, n_hidden, n_output, n_steps, dropout, activation = tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        # self.n_input = n_input
        self.n_hidden = 256
        # self.n_output = n_output
        self.n_steps = 15

        self.cnnx = cnn2d(n_input, n_hidden, 1024, n_steps, dropout)
        self.rnnx = rnn2d(n_input, 256, n_output, n_steps, dropout)

        self.x = tf.placeholder("float", [None, 15 * n_input])
        self.t = tf.placeholder("float", [None, n_output])
        self.y = self._conv_rec_net()

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.mse = tf.reduce_mean(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _conv_rec_net(self):
        # Reshape input picture
        ''''''
        x = tf.reshape(self.x, shape=[-1, 4, 101, 101])
        x = tf.transpose(x, [0, 2, 3, 1])

        # Convolution Layer
        conv1 = self.cnnx._conv2d(x, self.cnnx.weights['wc1'], self.cnnx.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.cnnx._maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.cnnx._conv2d(conv1, self.cnnx.weights['wc2'], self.cnnx.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.cnnx._maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.cnnx.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.cnnx.weights['wd1']), self.cnnx.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.cnnx.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.cnnx.weights['out']), self.cnnx.biases['out'])
        out = tf.nn.relu(out)
        # return out

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        out = tf.reshape(out, [-1, self.n_steps, 1024])
        x = tf.unstack(out, self.n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        out = tf.matmul(outputs[-1], self.rnnx.weights['out']) + self.rnnx.biases['out']
        # out = tf.nn.relu(out)
        return tf.nn.relu(out)


    def partial_fit(self, X, T):
        mse, opt = self.sess.run((self.mse, self.optimizer), feed_dict={self.x:X, self.t:T})
        return mse

    def predict(self, X, T):
        mse, valid = self.sess.run((self.mse, self.y), feed_dict={self.x:X, self.t:T})
        return mse, valid

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.y, feed_dict={self.x:X})

    def getWeights(self):
        return self.sess.run(self.weights['out'])

    def getBiases(self):
        return self.sess.run(self.weights['out'])

    def saveModel(self, i):
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/model_cnn_rnn_{}.ckpt".format(i))

    def loadModel(self, i):
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/model_cnn_rnn_{}.ckpt".format(i))


def crnn_train(mode, crnn, sample_dict, iter):
    batch = 0
    ''''''
    for index, features, labels in tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=100):
        ''''''
        for j in range(32):
            sample_index = np.random.choice(features.shape[0], 100)
            batch_xs = features[sample_index, :]
            batch_ys = labels[sample_index, :]
            # new_features = tcdpsm.tranform_feature(batch_xs, 15, shape=(-1, n_input))
            # new_labels = tcdpsm.duplicate_label(batch_ys, 15)
            rmse = np.sqrt(crnn.partial_fit(batch_xs, batch_ys))
            print('iter: %d, batch: %d, epoch: %d, rmse: %f'%(iter + 1, batch + 1, j + 1, rmse))
            ''''''
            # break
        batch += 1
        ''''''
        break

    crnn.saveModel(iter + 1)


def crnn_valid(mode, crnn, sample_dict, iter):
    batch = 0
    ori_labels, pre_labels = np.zeros((0, 1)), np.zeros((0, 1))
    ''''''
    for index, features, labels in tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=100):
        # _features = tcdpsm.tranform_feature(features, 15, shape=(-1, n_input))
        # _labels = tcdpsm.duplicate_label(labels, 15)
        _cost_, _labels_ = crnn.predict(features, labels)
        # _labels_ = np.reshape(np.reshape(_labels_, (-1, 15)).mean(axis=1), (-1, 1))
        pre_labels = np.concatenate((pre_labels, _labels_), axis=0)
        ori_labels = np.concatenate((ori_labels, labels), axis=0)

        print('iter: %d, batch: %d, epoch: %d, rmse: %f'%(iter + 1, batch + 1, 1, np.sqrt(_cost_)))
        batch += 1
        ''''''
        break

    return ori_labels, pre_labels


if __name__ == '__main__':
    n_input = 4 * 101 * 101
    n_hidden = 1024
    n_output = 1
    n_steps = 15
    dropout = 0.75
    crnn = cnnxrnn(n_input, n_hidden, n_output, n_steps, dropout)
    print('a cnn_x_rnn model is created! ')


    for i in range(128):
        sample_dict = tcdp.random_select_samples()

        print('training ... ')
        crnn_train('train', crnn, sample_dict, i)

        print('validating ... ')
        ys_train = crnn_valid('train', crnn, sample_dict, i)
        ys_valid = crnn_valid('valid', crnn, sample_dict, i)
        ys_testA = crnn_valid('testA', crnn, sample_dict, i)

        obj = {'train': ys_train, 'valid': ys_valid, 'testA': ys_testA}
        fn2 = 'ys_cnn_x_rnn_model_%d.cpk' % (i + 1)
        loader.saveObject(obj, fn2)

        ''''''
        break