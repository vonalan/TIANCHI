# coding=utf-8
# Author: kingdom

from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import utilities as utils
import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp
import tianchi_data_processor_for_seperate_model as tcdpsm
import tianchi_model_trainer_validator as tctv

# import tianchi_cnn_glm_model as tc_cnn_glm
# import tianchi_rnn_model as tc_rnn
import tianchi_glm_model as tc_glm
# import tianchi_mlpnn_model as tc_mlpnn
import tianchi_leastsq_model as tc_lsq


# split_dict_train = tcdpsm.load_splited_selected_samples()

# i don't know how to create rnn list gracefully
# rnns = [tc_rnn.rnn2d() for i in range(len(split_dict_train))]

# inmode = 2
# re_sample_dict = tcdpsm.load_splited_selected_samples()


'''bug bug bug'''
# weights = tc_lsq.pre_train(mini_batch_size=5000)
# print(weights)


model = tc_glm.glm2d()
model.name = 'fv_glm'
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
    tctv.train('train', str(0), model, sample_dict['train'], iter, mini_batch_size=4000)
    model.saveModel(str(0), iter)

    '''bug bug bug'''
    # ys_train = tctv.valid('train', str(0), model, sample_dict['train'], iter, mini_batch_size=1000)
    ys_valid = tctv.valid('train', str(0), model, sample_dict['valid'], iter, mini_batch_size=2000)
    ys_testA = tctv.valid('testA', str(0), model, sample_dict['testA'], iter, mini_batch_size=2000)

    model.saveResults('ys_valid', iter, ys_valid)
    model.saveResults('ys_testA', iter, ys_testA)

    '''bug bug bug'''
    # break