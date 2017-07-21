# coding=utf-8
# Author: kingdom

from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import rnn

import utilities as utils
import tianchi_rnn_model as tc_rnn 
import tianchi_model_trainer_validator as tctv


model = tc_rnn.rnn2d()
model.name = 'rnn_ori'
model.eta = 0.1
# print(model.get_weights('f1'))


'''bug bug bug'''
# check_point = 2
# model.loadModel(0, check_point)
# model.eta = 0.1 * pow(0.9, check_point + 1)


sample_dict = tcdp.random_select_samples()


cbegin = 0
cend = 10
for iter in range(cbegin, cend):

    '''bug bug bug'''
    tctv.train('train', str(0), model, sample_dict['train'], iter, mini_batch_size=500)
    model.saveModel(str(0), iter)

    '''bug bug bug'''
    # ys_train = tctv.valid('train', str(0), model, sample_dict['train'], iter, mini_batch_size=1000)
    ys_valid = tctv.valid('train', str(0), model, sample_dict['valid'], iter, mini_batch_size=500)
    ys_testA = tctv.valid('testA', str(0), model, sample_dict['testA'], iter, mini_batch_size=500)

    model.saveResults('ys_valid', iter, ys_valid)
    model.saveResults('ys_testA', iter, ys_testA)

    '''bug bug bug'''
    # break