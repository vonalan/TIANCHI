import os

import numpy as np
from sklearn.metrics import mean_squared_error as sklmse
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn.preprocessing as sklpreprocessor
from sklearn.decomposition import PCA as sklpca

# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model as tcdpsm


class LSQ(object):
    def __init__(self):
        self.name = ''
        # self.mode = ''

        self.len_traj = 30720
        self.len_hof = 110592
        self.len_hog = 98304
        self.len_mbhx = 98304
        self.len_mbhy = 98304

        # self.len_pca = len_pca

        '''bug bug bug'''
        # self.n_input = self.len_traj + self.len_hof + self.len_hog + self.len_mbhx + self.len_mbhy
        # self.n_input = self.len_traj + 1
        # self.n_input = self.len_hof + self.len_hog
        # self.n_input = self.len_mbhx + self.len_mbhy

        self.n_input = self.len_traj

        # self.n_hidden = 1024
        self.n_output = 1


        self.weights = np.zeros((self.n_input, self.n_output))
        self.biases = np.zeros((self.n_output,1))

        self.func = self._function


    def _function(self, X, Y):
        pass

    def train(self, X_, Y_):
        print('training ... ')
        self.weights = np.dot(np.linalg.pinv(X_), Y_)
        Y = np.dot(X_, self.weights)
        rmse = np.sqrt(sklmse(Y_, Y))

        return Y, rmse


    def predict(self, X_, Y_):
        print('validating ... ')
        Y = np.dot(X_, self.weights)
        rmse = np.sqrt(sklmse(Y_, Y))

        return Y, rmse

    def saveResults(self, prefix, iter, results):
        np.savetxt("../data/%s/%s_%s_%d.txt" % (self.name, prefix, self.name, iter), results)


def pre_train(mini_batch_size=1000):
    lsq = LSQ()
    '''bug bug bug'''
    sample_dict = tcdp.random_select_samples()

    mode = 'train'
    for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=mini_batch_size)):
        y, rmse = lsq.train(features, labels)
        # np.savetxt('ys_train_lsq.txt', np.hstack((index, labels, y)))
        print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
        break

    mode = 'valid'
    for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches('train', sample_dict[mode], mini_batch_size=2000)):
        y, rmse = lsq.predict(features, labels)
        # np.savetxt('ys_valid_lsq.txt', np.hstack((index, labels, y)))
        print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
        break

    # mode = 'testA'
    # for _, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict[mode], mini_batch_size=2000)):
    #     y, rmse = lsq.predict(features, labels)
    #     np.savetxt('ys_testA_lsq.txt', np.hstack((index, labels, y)))
    #     print(y.tolist()[-6:], '\n', labels.tolist()[-6:], '\n', rmse)
    #     break
    #
    # print('\n')

    return (lsq.weights).astype(np.float32)


if __name__ == '__main__':
    _ = pre_train(mini_batch_size=10000)