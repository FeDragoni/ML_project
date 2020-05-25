import numpy as np
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
import gen_dataset
from sklearn.svm import SVR
import csv
from sklearn.model_selection import GridSearchCV
import Functions
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll

# Data pre-processing
train_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TRAIN.csv', shuf=True, delimiter=',')
test_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TEST.csv', shuf=True, delimiter=',')
x_train,y_train=gen_dataset.divide(train_dataset)
x_test,y_test=gen_dataset.divide(test_dataset)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

param_grid_RBF = {
  'estimator__C':[x for x in np.logspace(start = np.log10(1), stop = np.log10(100), num = 10)] ,
  'estimator__gamma':[x for x in np.logspace(start = np.log10(0.05), stop = np.log10(0.3), num = 10)] ,
  'estimator__epsilon' : [0.1]
}
parameters_rbf_2 = {
  'gamma':[x for x in np.logspace(start = np.log10(0.05), stop = np.log10(0.3), num = 10)],
  'C':[x for x in np.linspace(start = 10, stop = 30, num = 10)]}

param_dist_BO = {
  'kernel': hp.choice('kernel', ['poly']),
  'degree': hp.choice('degree', [3]),
  #'coef0': hp.choice('coef0', [0,1.5,3]),
  'gamma': hp.loguniform('gamma', np.log(0.01), np.log(0.5)),
  'C': hp.loguniform('C', np.log(1e-1), np.log(1e1))
}


param_grid_POLY = {
  'estimator__C':[x for x in np.logspace(start = np.log10(0.005), stop = np.log10(0.5), num = 6)] ,
  'estimator__gamma': [x for x in np.logspace(start = np.log10(0.01), stop = np.log10(0.45), num = 6)] ,
  'estimator__epsilon' : [0.1],
  'estimator__degree' : [3],
  'estimator__coef0' : [1.5]
}

gb = svm.SVR(kernel= 'poly')
grid_fitted = Functions.hp_tuning_svm_regr_GS(gb, x_train, y_train, param_grid_POLY, filename= 'train_result_poly.csv' )

# Functions.hp_tuning_svm_regr_GS(gb, x_train, y_train,param_grid_RBF, filename= 'prova.csv' )
# Functions.hp_tuning_svm_regr_RS(gb, x_train, y_train,param_grid )
# Functions.hp_tuning_BO(gb, x_train, y_train,param_dist_BO )

best_one = svm.SVC(**grid_fitted.best_params_)
best_one.fit (x_train, y_train)
print (best_one.score (x_test, y_test))

# Test score
mySVM = svm.SVR(kernel='rbf', C = 18.32   , gamma =  0.135 , epsilon=0.1, degree = 3)
myRegr = MultiOutputRegressor(mySVM)
myRegr.fit (x_train, y_train)
y_pred = myRegr.predict(x_test)
y_pred_train = myRegr.predict(x_train)
print('TEST')
print('MSE: ',sklearn.metrics.mean_squared_error(y_test,y_pred))
print('MAE: ',sklearn.metrics.mean_absolute_error(y_test,y_pred))
print('R^2: ',myRegr.score (x_test, y_test))
MEE = 0
for pred_val, actual_val in zip(y_pred, y_test):
    MEE += np.linalg.norm(pred_val - actual_val)
MEE = MEE/(y_train.shape[0])
print('MEE: ', MEE)
# Train score
print('TRAIN')
print('MSE_train: ',sklearn.metrics.mean_squared_error(y_train,y_pred_train))
print('MAE_train: ',sklearn.metrics.mean_absolute_error(y_train,y_pred_train))
print('R^2_train: ',myRegr.score (x_train, y_train))
MEE_train = 0
for pred_val, actual_val in zip(y_pred_train, y_train):
    MEE_train += np.linalg.norm(pred_val - actual_val)
MEE_train = MEE_train/(y_train.shape[0])
print('MEE_train: ', MEE_train)
#########################################################à