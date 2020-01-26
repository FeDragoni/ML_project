import numpy as np
import itertools
import pandas as pd
# import datacontrol
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import time
import gen_dataset
# import validation
import csv
import time
import scipy.stats as stats

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
def hp_tuning_svm_GS(svm, x, y, param_grid, folds=5, save=True, filename="SVM_GS.csv"):
	start_time = time.time()
	y=np.ravel(y,order='C')
	clf = GridSearchCV(svm, param_grid, cv=folds)
	grid_fitted = clf.fit(x, y)
	print("Time used for grid search: %.3f" %(time.time()-start_time))
	means = grid_fitted.cv_results_['mean_test_score']
	stds = grid_fitted.cv_results_['std_test_score']
	params = grid_fitted.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
	if save:
		df=pd.DataFrame(zip(means, params))
		df = df.rename(index=str, columns={0: "Mean Validation Error", 1: "Parameters"})
		df.to_csv("./result/"+filename)
	print("Total elapsed time: %.3f" %(time.time()-start_time))
	return grid_fitted

def hp_tuning_svm_RS(svm, x, y, param_dist, iterations=10, folds=5, save=True, filename="SVM_RS.csv"):
	# NOTE: continuous parameters should be given as a distribution for a proper random search
    # Distributions can be generated with scipy.stats module
    # For parameters that need to be explored in terms of order of magnitude loguniform distribution is recommended
	start_time = time.time()
	y=np.ravel(y,order='C')
	clf = RandomizedSearchCV(svm, param_dist, n_iter=iterations, cv=folds)
	grid_fitted = clf.fit(x, y)
	print("Time used for randomized search: %.3f" %(time.time()-start_time))
	means = grid_fitted.cv_results_['mean_test_score']
	stds = grid_fitted.cv_results_['std_test_score']
	params = grid_fitted.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
	if save:
		df=pd.DataFrame(zip(means, params))
		df = df.rename(index=str, columns={0: "Mean Validation Error", 1: "Parameters"})
		df.to_csv("./result/"+filename)
	print("Total elapsed time: %.3f" %(time.time()-start_time))
	return grid_fitted

param_dist = {'kernel': ['linear', 'rbf', poly],
              'C': stats.loguniform(1e-4, 1e0)}

svc = svm.SVC()
hp_tuning_svm_RS(svc,x_train,y_train,param_dist)



