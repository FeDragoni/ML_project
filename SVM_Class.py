import numpy as np
import itertools
import pandas as pd
# import datacontrol
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
import gen_dataset
# import validation
import csv
from sklearn.model_selection import GridSearchCV
# import time
import Functions
import scipy.stats as stats
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll

import Functions
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import scipy as sp

##SVM Classifier
def encode_MONK(x):
	ohe = OneHotEncoder()
	x = ohe.fit_transform(x).toarray()
	return x

train_dataset = pd.read_csv('./Monk_dataset/monks-3.train',delimiter=' ')
x_train = train_dataset.iloc[:, 2:8].values
y_train = train_dataset.iloc[:, 1].values
y_train = y_train.reshape(y_train.shape[0], 1)

test_dataset = pd.read_csv('./Monk_dataset/monks-3.test',delimiter=' ')
x_test = test_dataset.iloc[:, 2:8].values
y_test = test_dataset.iloc[:, 1].values
y_test = y_test.reshape(y_test.shape[0], 1)

x_train = encode_MONK(x_train)
x_test = encode_MONK(x_test)

print(y_train.shape)
print(x_train.shape)
x_train = np.asarray(x_train,dtype=np.float64)
y_train = np.asarray(y_train,dtype=np.float64)

print (y_train.shape)

###kernel = rbf
parameters_rbf = {
    'gamma':[0.01, 0.1, 1, 10, 100, 1000],
    'C': [0.01, 0.1, 1, 10, 100, 1000] }
parameters_rbf_2 = {
    'gamma':[x for x in np.logspace(start = np.log10(0.001), stop = np.log10(1), num = 50)],
    'C':[x for x in np.linspace(start = 0.1, stop = 100, num = 50)]
}

###kernel = poly
parameters_poly = {
    'gamma':[0.01, 0.1, 1, 10, 100, 1000], 
    'C': [0.01, 0.1, 1, 10, 100, 1000] ,
    'degree' : [1,2,3],
    'coef0' : [1,2,3]
    }
parameters_poly_2 = {
    'gamma':[x for x in np.logspace(start = np.log10(0.025), stop = np.log10(0.4), num = 20)],
    'C': [x for x in np.logspace(start = np.log10(5), stop = np.log10(300), num = 20)], 
    'degree' : [1,2,3],
    'coef0' : [0,1.5,3 ]
    }
param_dist_BO = {
    'kernel': hp.choice('kernel', ['poly']),
    'degree': hp.choice('degree', [3]),
    'coef0': hp.choice('coef0', [0,1.5,3]),
    'gamma': hp.loguniform('gamma', np.log(0.01), np.log(10)),
    'C': hp.loguniform('C', np.log(1e-1), np.log(1e2))
}
svc = svm.SVC(kernel='poly')


grid_fitted = Functions.hp_tuning_GS( svc, x_train, y_train,parameters_poly_2, folds=5, save=True, filename="GS_poly_run_mattina_1.csv")
# grid_fitted = Functions.hp_tuning_RS( svc, x_train, y_train,  parameters_poly_2, iterations=200, folds=5, save=True, filename="M1_RS_poly_run5.csv",)
# Functions.hp_tuning_BO( svm.SVC, x_train, y_train, param_dist_BO ,iterations=200, save=True, filename="M2_BO_poly_run1.csv",)
best_one = svm.SVC(**grid_fitted.best_params_)
# best_one = svm.SVC(C = 11.839   , gamma =  0.025 , degree = 2, coef0 = 3, kernel='poly')
# best_one.fit (x_train, np.ravel(y_train,order='C'))

# print (best_one.score (x_test, y_test))
mySVM= svm.SVC(C = 11.839   , gamma =  0.025 , degree = 2, coef0 = 3, kernel='poly')
mySVM.fit (x_train, np.ravel(y_train,order='C'))
print ("Train accuracy: ",mySVM.score (x_train, y_train))
print ("Test accuracy: ",mySVM.score (x_test, y_test))
