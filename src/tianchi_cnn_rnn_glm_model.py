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


class lgm2d(object): 
    def __init__(self, activation = tf.nn.softplus):
        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101

        self.n_cnn_glm = 1
        self.n_cnn_rnn = 1024
        self.n_rnn_lgm = 1

        self.n_hidden = 128
        self.n_input = self.n_steps * self.n_height
        self.n_output = 1

        self.dropout = 0.75
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_input, self.n_output]))
        }
        return weights

    def _initialize_biases(self):
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }
        return biases


class cnn2d(object):
    def __init__(self, activation = tf.nn.relu):
        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101

        self.n_cnn_glm = 1
        self.n_cnn_rnn = 1024
        self.n_rnn_lgm = 1

        self.n_hidden = 1024
        self.n_input = self.n_width * self.n_length
        self.n_output = self.n_cnn_glm

        self.dropout = 0.75
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def _initialize_weights(self):
        weights = {
            ''''''
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([26 * 26 * 64, 1024])),
            ''''''
            'out': tf.Variable(tf.random_normal([1024, self.n_cnn_glm]))
        }
        return weights

    def _initialize_biases(self):
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            ''''''
            'out': tf.Variable(tf.random_normal([self.n_cnn_glm]))
        }
        return biases


class rnn2d(object):
    def __init__(self, activation = tf.nn.relu):
        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101
        self.n_channel = 1

        self.n_cnn_rnn = 1024
        self.n_rnn_lgm = 1

        self.n_hidden = 128
        self.n_input = self.n_width * self.n_length
        ''''''
        self.n_output = 1

        self.dropout = 0.75
        self.activation = activation

        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
        }
        return weights

    def _initialize_biases(self):
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }
        return biases


class cnn_rnn_glm(object):
    def __init__(self, activation = tf.nn.relu):
        self.n_steps = 15
        self.n_height = 4
        self.n_width = 101
        self.n_length = 101
        self.n_channel = 1

        self.n_cnn_glm = 1
        self.n_cnn_rnn = 1024
        self.n_rnn_lgm = 1

        self.n_hidden = 128
        self.n_input = self.n_steps * self.n_height * self.n_width * self.n_length * self.n_channel
        self.n_output = 1

        self.eta = 0.1
        self.dropout = 0.75
        self.activation = activation

        self.cnnx = cnn2d()
        # self.rnnx = rnn2d()
        self.glmx = lgm2d()

        self.x = tf.placeholder("float", [None, self.n_input])
        self.t = tf.placeholder("float", [None, self.n_output])
        self.y = self._conv_rec_glm_net()

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.t), 2.0))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.y, self.t), 2.0)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _conv_rec_glm_net(self):
        # Reshape input picture
        ''''''
        x = tf.reshape(self.x, shape=[-1, self.n_width, self.n_length, self.n_channel])
        # x = tf.transpose(x, [0, 2, 1, 3, 4, 5])

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

        # Output,
        out = tf.add(tf.matmul(fc1, self.cnnx.weights['out']), self.cnnx.biases['out'])
        out = tf.nn.relu(out)
        # return out

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # ''''''
        # out = tf.reshape(out, [-1, self.n_steps, self.n_height, self.n_cnn_rnn])
        # out = tf.transpose(out, [0, 2, 1, 3])
        # out = tf.reshape(out, [-1, self.n_steps, self.n_cnn_rnn])
        # ''''''
        # x = tf.unstack(out, self.n_steps, 1)

        # # Define a lstm cell with tensorflow
        # lstm_cell = rnn.BasicLSTMCell(self.rnnx.n_hidden, forget_bias=1.0)

        # # Get lstm cell output
        # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # # Linear activation, using rnn inner loop last output
        # out =  tf.matmul(outputs[-1], self.rnnx.weights['out']) + self.rnnx.biases['out']
        # out = tf.nn.relu(out)

        # Get GLM output
        out = tf.reshape(out, [-1, self.n_cnn_glm * self.n_steps * self.n_height])
        out = tf.matmul(out, self.glmx.weights['out']) + self.glmx.biases['out']
        out = tf.nn.relu(out)

        return out 


    def partial_fit(self, X, T):
        rmse, opt = self.sess.run((self.rmse, self.optimizer), feed_dict={self.x:X, self.t:T})
        return rmse

    def predict(self, X, T):
        rmse, valid = self.sess.run((self.rmse, self.y), feed_dict={self.x:X, self.t:T})
        return rmse, valid

    def saveModel(self, i):
        if not os.path.exists('../data/cnn_glm_model/'): 
            os.mkdir('../data/cnn_glm_model/')
        saver = tf.train.Saver()
        saver.save(self.sess, "../data/cnn_glm_model/cnn_glm_model_{}.ckpt".format(i))

    def loadModel(self, i):
        saver = tf.train.Saver()
        saver.restore(self.sess, "../data/cnn_glm_model/cnn_glm_model_{}.ckpt".format(i))


def train(mode, inmode, crlnn, sample_dict, iter):
    threshold = 12.00
    ''''''
    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=50)):
        crlnn.eta = crlnn.eta * 0.99

        list_rmse = 8 * [threshold * 10]
        indicator = threshold * 10

        epoch = 0
        while indicator >= threshold:
            sample_index = np.random.choice(features.shape[0], 50)
            batch_xs = features[sample_index, :]
            batch_ys = labels[sample_index, :]
            rmse = crlnn.partial_fit(batch_xs, batch_ys)
            list_rmse[epoch%8] = rmse

            if (epoch + 1) % 2 == 0:
                indicator = sum(list_rmse)/len(list_rmse)
                print('mode: %s, inmode: %s, iter: %d, batch: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f'% \
                     (mode, inmode, iter + 1, batch + 1, epoch + 1, crlnn.eta, rmse, indicator))

            epoch += 1
        ''''''
        # re_sample_dict = tcdp.re_random_select_samples(sample_dict['valid'][inmode],50)
        _, _ = valid('valid', inmode, crlnn, sample_dict, iter, memo = 'train') 

        ''''''
        break

    # crlnn.saveModel(mode, inmode, iter)



def valid(mode, inmode, crlnn, sample_dict, iter, memo = ''):
    ori_labels, pre_labels = np.zeros((0, 1)), np.zeros((0, 1))
    ''''''
    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=50)):
        _rmse_, _labels_ = crlnn.predict(features, labels)
        pre_labels = np.concatenate((pre_labels, _labels_), axis=0)
        ori_labels = np.concatenate((ori_labels, labels), axis=0)

        print('mode: %s, inmode: %s, iter: %d, batch: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f\n'% \
             (mode, inmode, iter + 1, batch + 1, 1, crlnn.eta, _rmse_, 14.68))
        
        ''''''
        if memo == 'train': 
            break
        else:
            break 

    return ori_labels, pre_labels


if __name__ == '__main__':
    crlnn = cnn_rnn_glm()
    print('a cnn_x_rnn_x_glm model is created! ')

    crlnn.eta = 0.01 
    for i in range(128):
        # sample_dict = tcdp.random_select_samples()
        # re_sample_dict = tcdpsm.split_selected_samples(i + 1, sample_dict)
        re_sample_dict, _ = tcdpsm.load_split_selected_samples(i)

        print('training ... ')
        train('train', 'ge_20', crlnn, re_sample_dict['train']['ge_20'], i)
        train('train', 'lt_20', crlnn, re_sample_dict['train']['lt_20'], i)

        # print('validating ... ')
        # ys_train = crnn_valid('train', crnn, sample_dict, i)
        # ys_valid = valid('valid', crlnn, sample_dict, i)
        # ys_testA = valid('testA', crlnn, sample_dict, i)

        # obj = {'train': valid, 'valid': ys_valid, 'testA': ys_testA}
        # obj = {'valid': ys_valid, 'testA': ys_testA}
        # fn2 = 'ys_cnn_x_rnn_model_%d.cpk' % (i + 1)
        # loader.saveObject(obj, fn2)

        ''''''
        break
