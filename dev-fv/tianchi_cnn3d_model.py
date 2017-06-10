'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


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

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.t = tf.placeholder(tf.float32, [None, self.n_output])
        self.y = self._conv_net()

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.optimizer = optimizer.minimize(self.cost)

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

    def _conv_net(self):
        # Reshape input picture
        ''''''
        x = tf.reshape(self.x, shape=[-1, 4, 101, 101])
        x = tf.transpose(x, [0, 2, 3, 1])

        # Convolution Layer
        conv1 = self._conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self._maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self._conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self._maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

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

    def partial_fit(self, X, T):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x:X, self.t:T})
        return cost

    def predict(self, X, T):
        cost, valid = self.sess.run((self.cost, self.y), feed_dict={self.x:X, self.t:T})
        return cost, valid


    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.y, feed_dict={self.x:X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['out'])

    def getBiases(self):
        return self.sess.run(self.weights['out'])

    def saveModel(self, i):
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/model_cnn_{}.ckpt".format(i))

    def loadModel(self, i):
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/model_cnn_{}.ckpt".format(i))


def cnn2d_train(mode, cnn, sample_dict, iter):
    batch = 0
    ''''''
    for index, features, labels in tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=100):
        ''''''
        for j in range(32):
            sample_index = np.random.choice(features.shape[0], 100)
            batch_xs = features[sample_index, :]
            batch_ys = labels[sample_index, :]
            new_features = tcdpsm.tranform_feature(batch_xs, 15, shape=(-1, n_input))
            new_labels = tcdpsm.duplicate_label(batch_ys, 15)
            rmse = np.sqrt(cnn.partial_fit(new_features, new_labels))
            print('iter: %d, batch: %d, epoch: %d, rmse: %f'%(iter + 1, batch + 1, j + 1, rmse))
            ''''''
            # break
        batch += 1
        ''''''
        # break

    cnn.saveModel(iter + 1)


def cnn2d_valid(mode, cnn, sample_dict, iter):
    batch = 0
    ori_labels, pre_labels = np.zeros((0, 1)), np.zeros((0, 1))
    ''''''
    for index, features, labels in tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=100):
        _features = tcdpsm.tranform_feature(features, 15, shape=(-1, n_input))
        _labels = tcdpsm.duplicate_label(labels, 15)

        _cost_, _labels_ = cnn.predict(_features, _labels)
        _labels_ = np.reshape(np.reshape(_labels_, (-1, 15)).mean(axis=1), (-1, 1))
        pre_labels = np.concatenate((pre_labels, _labels_), axis=0)
        ori_labels = np.concatenate((ori_labels, labels), axis=0)

        print('iter: %d, batch: %d, epoch: %d, rmse: %f'%(iter + 1, batch + 1, 1, np.sqrt(_cost_)))
        batch += 1
        ''''''
        # break

    return ori_labels, pre_labels


if __name__ == '__main__':
    n_input = 4 * 101 * 101
    n_hidden = 1024
    n_output = 1
    n_steps = 15
    dropout = 0.75
    cnn = cnn2d(n_input, n_hidden, n_output, n_steps, dropout)
    print('a cnn model is created! ')


    for i in range(128):
        sample_dict = tcdp.random_select_samples()

        print('training ... ')
        cnn2d_train('train', cnn, sample_dict, i)

        print('validating ... ')
        ys_train = cnn2d_valid('train', cnn, sample_dict, i)
        ys_valid = cnn2d_valid('valid', cnn, sample_dict, i)
        ys_testA = cnn2d_valid('testA', cnn, sample_dict, i)

        obj = {'train': ys_train, 'valid': ys_valid, 'testA': ys_testA}
        fn2 = 'ys_cnn_model_%d.cpk' % (i + 1)
        loader.saveObject(obj, fn2)

        ''''''
        # #break