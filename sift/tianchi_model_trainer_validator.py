from __future__ import print_function

import os 
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def load(mode, inmode, f, k):
    I = np.loadtxt('../data/common/index_%s.txt'%mode)
    Y = np.loadtxt('../data/common/label_%s.txt'%mode)
    X = np.loadtxt('../data/hist_%s_%s_f%d_k%d.txt'%(mode, inmode, f, k))
    
    return I, Y, X


def train(mode, inmode, model, iter, mini_batch_size = 1000):
    '''bug bug bug'''
    epoches = 32
    threshold = 12.04
    show_step = 8
    
    _, labels, features = load('train', 'std', 10, 128)

    list_rmse = epoches * [threshold * epoches]
    indicator = threshold * epoches

    epoch = 0
    while indicator >= threshold:
        sample_index = np.random.choice(features.shape[0], mini_batch_size)
        batch_xs = features[sample_index, :]
        batch_ys = labels[sample_index, :]
        rmse = model.partial_fit(batch_xs, batch_ys)
        list_rmse[epoch%epoches] = rmse

        if (epoch + 1) % show_step == 0:
            indicator = sum(list_rmse)/len(list_rmse)
            print('mode: %s, inmode: %s, iter: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f'% \
                    (mode, inmode, iter + 1, epoch + 1, model.eta, rmse, indicator))

        epoch += 1

        '''bug bug bug'''
        if epoch > 1024:
            break
    
    '''bug bug bug'''
    # break


def valid(mode, inmode, model, sample_dict, iter, mini_batch_size = 1000, memo = ''):
    # ori_index, ori_labels, pre_labels = np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

    index, labels, features = load('trainB', 'std', 10, 128)
    _rmse, _labels = model.predict(features, labels)

    print('mode: %s, inmode: %s, iter: %d, batch: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f'% \
            (mode, inmode, iter + 1, 1, model.eta, _rmse, 14.68))
    print(_labels.tolist()[-5:], '\n', labels.tolist()[-5:], '\n')

    '''bug bug bug'''
    # break

    return np.hstack((index, labels, _labels))