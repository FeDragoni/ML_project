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
##SVM Classifier

##creo dataset
train_array = gen_dataset.CSV_to_array('./Monk_dataset/monks-1.train', shuf = True, delimiter = ' ')
test_array = gen_dataset.CSV_to_array('./Monk_dataset/monks-1.test', shuf = True, delimiter = ' ')
x_train = train_array[:,1:7]
y_train = train_array[:,0]
y_train = y_train.reshape(y_train.shape[0],1)
x_test = test_array[:,1:7]
y_test = test_array[:,0]
y_test = y_test.reshape(y_test.shape[0],1)

x_train = np.asarray(x_train,dtype=np.float64)
y_train = np.asarray(y_train,dtype=np.float64)

print (y_train.shape)

##parametri

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()

clf = GridSearchCV(svc, parameters)
grid_fitted = clf.fit(x_train, np.ravel(y_train,order='C'))
means = grid_fitted.cv_results_['mean_test_score']
stds = grid_fitted.cv_results_['std_test_score']
params = grid_fitted.cv_results_['params']
array_kernel = []
array_C = []

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    array_kernel = array_kernel + [param['kernel']]
    array_C = array_C +  [mean]

print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
print (array_kernel)
print (array_C)
