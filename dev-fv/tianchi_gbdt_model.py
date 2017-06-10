import random 

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp


def train(mode, inmode, model, sample_dict, mini_batch_size=100):
    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=mini_batch_size)):
        model.fit(features, labels) 

        pre_labs = model.predict(features).reshape(-1,1)
        rmse = np.sqrt(mean_squared_error(labels, pre_labs))
        ys = np.hstack((index, labels, pre_labs))

        print(mode, inmode, rmse)
        return ys

def valid(mode, inmode, model, sample_dict, mini_batch_size=100):
    for batch, (index, features, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_dict, mini_batch_size=mini_batch_size)):
        pre_labs = model.predict(features).reshape(-1,1)
        rmse = np.sqrt(mean_squared_error(labels, pre_labs))
        ys = np.hstack((index, labels, pre_labs))

        print(mode, inmode, rmse)
        return ys
  

sample_dict = tcdp.random_select_samples()

n_est = [1024]
eta = [0.01]
depth = [2]

for n in n_est:
    for e in eta:
        for d in depth:
            model = GradientBoostingRegressor(n_estimators=n, learning_rate=e, max_depth=d, random_state=0, loss='ls')
            model_name = 'gbdt_n%d_e%f_d%d'%(n, e, d)

            ys_train = train('train', [n, e, d], model, sample_dict['train'], mini_batch_size=2000)
            ys_valid = valid('train', [n, e, d], model, sample_dict['valid'], mini_batch_size=2000)
            # ys_testA = valid('testA', model, sample_dict['testA'], mini_batch_size=2000)

            # np.savetxt('ys_train_%s.txt'%model_name, ys_train)
            # np.savetxt('ys_valid_%s.txt'%model_name, ys_valid)
            # np.savetxt('ys_testA_%s.txt'%model_name, ys_testA)
