# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
#这个cross_validation模块取消了，但是这个函数还能用，把cross_validation改成model_selection就行

# load or create your dataset

print('Load data...')
total_data = pd.read_csv('processed_total_data.csv')
print (total_data)

# 说明一下，这个total_data_X是所有的特征列，就是X；然后total_data_Y就是label列。但是这个是所有的，下边划分验证集和训练集
# total_data_X=total_data.iloc[:,0:-1]
# total_data_Y=total_data.iloc[:,-1]
# print (total_data_X)
# print (total_data_Y)

X_train,X_test,y_train,y_test=train_test_split(total_data.iloc[:,0:-1],total_data.iloc[:,-1],test_size=0.2,random_state=0)

# print (X_train)
# print (X_test)
# print (y_train)
# print (y_test)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')


print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print (y_pred)
np.savetxt("y_pred.csv",y_pred)
# eval
print('The rmse of prediction i s:', mean_squared_error(y_test, y_pred) ** 0.5)