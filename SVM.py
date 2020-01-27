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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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
		df.to_csv(path_or_buf=("./result/"+filename),sep=',')
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
		df.to_csv(path_or_buf=("./result/"+filename),sep=',', index_label='Index')
	print("Total elapsed time: %.3f" %(time.time()-start_time))
	return grid_fitted

def bayesian_func_generator_classification(x,y,n_splits=5):
	def score_func (param_dist):
		svc = svm.SVC(**param_dist)
		score = cross_val_score(svc, x, y,cv=n_splits).mean()
		return {'loss': (-score), 'status': STATUS_OK}
	return score_func

def hp_tuning_svm_BO(svm,x,y,param_dist,iterations=10,):
	y=np.ravel(y,order='C')
	objective_function = bayesian_func_generator_classification(x,y)
	trials = Trials()
	best_param = fmin(objective_function, 
                  param_dist, 
                  algo=tpe.suggest, 
                  max_evals=iterations, 
                  trials=trials,
                  rstate=np.random.RandomState(1)
				  )
	best_param_values = [val for val in best_param.values()]
	losses = [x['result']['loss'] for x in trials.trials]
	vals = [x['misc']['vals']for x in trials.trials]
	for val, loss in zip(vals,losses):
		print('Score: %f   Param:%s' %(loss,val))
	best_param_values = [x for x in best_param.values()]
	print("Best loss obtained: %f\n with parameters: %s" % (-min(losses), best_param_values))
	return trials


param_dist_RS = {'kernel': ['linear', 'rbf', 'poly'],
              'C': stats.loguniform(1e-4, 1e0)}

param_dist_BO = {
    'C': hp.loguniform('C', np.log(1e-4), np.log(1e2)),
    #'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
    #'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
}

svc = svm.SVC()
#hp_tuning_svm_BO(svc,x_train,y_train,param_dist_BO,iterations=50)
hp_tuning_svm_RS(svc,x_train,y_train,param_dist_RS)



