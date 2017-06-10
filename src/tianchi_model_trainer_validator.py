from __future__ import print_function

import os 
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# import loader
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model as tcdpsm


def train(mode, inmode, model, sample_dict, iter, mini_batch_size = 1000):
    '''bug bug bug'''
    epoches = 32
    threshold = 12.04
    show_step = 8
    
    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=mini_batch_size)):
        model.eta = model.eta * 0.99

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
                print('mode: %s, inmode: %s, iter: %d, batch: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f'% \
                     (mode, inmode, iter + 1, batch + 1, epoch + 1, model.eta, rmse, indicator))

            epoch += 1

            '''bug bug bug'''
            if epoch > 1024:
                break
        
        '''bug bug bug'''
        # break


def valid(mode, inmode, model, sample_dict, iter, mini_batch_size = 1000, memo = ''):
    ori_index, ori_labels, pre_labels = np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=mini_batch_size)):
        _rmse, _labels = model.predict(features, labels)
        ori_index = np.concatenate((ori_index, index), axis = 0)
        pre_labels = np.concatenate((pre_labels, _labels), axis=0)
        ori_labels = np.concatenate((ori_labels, labels), axis=0)

        print('mode: %s, inmode: %s, iter: %d, batch: %d, epoch: %d, eta: %f, rmse: %f, indicator: %f'% \
             (mode, inmode, iter + 1, batch + 1, 1, model.eta, _rmse, 14.68))
        print(pre_labels.tolist()[-5:], '\n', ori_labels.tolist()[-5:], '\n')

        '''bug bug bug'''
        # break

    return np.hstack((ori_index, ori_labels, pre_labels))