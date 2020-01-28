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

##parametri
def hp_tuning_GS(estimator, x, y, param_grid, folds=5, save=True, filename="SVM_GS.csv"):
	start_time = time.time()
	y=np.ravel(y,order='C')
	clf = GridSearchCV(estimator, param_grid, cv=folds)
	grid_fitted = clf.fit(x, y)
	print("Time used for grid search: %.3f" %(time.time()-start_time))
	means_test = grid_fitted.cv_results_['mean_test_score']
	stds = grid_fitted.cv_results_['std_test_score']
	params = grid_fitted.cv_results_['params']
	for mean, stdev, param in zip(means_test, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
	if save:
			dict_csv={}
			dict_csv.update({'Score' : []})
			for key in params[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(params):
				for key in val:
					dict_csv[key].append((val[key]))
				dict_csv['Score'].append(means_test[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/"+filename),sep=',', index_label='Index')

	return grid_fitted

def hp_tuning_RS(estimator, x, y, param_dist, iterations=10, folds=5, save=True, filename="SVM_CLASS_RS.csv"):
	# NOTE: continuous parameters should be given as a distribution for a proper random search
    # Distributions can be generated with scipy.stats module
    # For parameters that need to be explored in terms of order of magnitude loguniform distribution is recommended
	start_time = time.time()
	y=np.ravel(y,order='C')
	clf = RandomizedSearchCV(estimator, param_dist, n_iter=iterations, cv=folds)
	grid_fitted = clf.fit(x, y)
	print("Time used for randomized search: %.3f" %(time.time()-start_time))
	means_test = grid_fitted.cv_results_['mean_test_score']
	stds = grid_fitted.cv_results_['std_test_score']
	params = grid_fitted.cv_results_['params']
	for mean, stdev, param in zip(means_test, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
	if save:
			dict_csv={}
			dict_csv.update({'Score' : []})
			for key in params[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(params):
				for key in val:
					dict_csv[key].append((val[key]))
				dict_csv['Score'].append(means_test[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/"+filename),sep=',', index_label='Index')
	return grid_fitted

def hp_tuning_svm_regr_GS(svr, x_train, y_train, param_grid, folds=5, save=True, filename="SVM_REGRESSOR_GS.csv"):
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_gamma = []
    array_C = []
    array_epsilon = []
    array_means = []
    # param = kwargs
    # print(param)
    # clf = GridSearchCV(gs_svr, param)
    print('\n\n\n\n')

    gs = GridSearchCV(MultiOutputRegressor(svr), param_grid=param_grid, return_train_score=True)
    gs_svr = gs.fit(x_train,y_train)
    grid_fitted = gs_svr
    # grid_fitted = clf.fit(x, y)
    means = grid_fitted.cv_results_['mean_test_score']
    means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        array_gamma = array_gamma + [param['estimator__gamma']]
        array_epsilon = array_epsilon + [param['estimator__epsilon']]
        array_C = array_C +  [param ['estimator__C']]
        array_means = array_means +  [mean]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [ array_C , array_gamma , array_epsilon,  array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    df=pd.DataFrame(array_tot)
    df = df.rename(index=str, columns={0: "Parameter_C", 1: "Epsilon" , 2: "Gamma" ,3: "Mean_Val_Error" })
    df.to_csv("./result/SVM_REGRESSOR_GS.csv")
    gh = pd.read_csv("./result/SVM_REGRESSOR_GS.csv")
    print(gh)

def hp_tuning_svm_regr_RS(svr, x_train, y_train, param_grid, folds=5, save=True, filename="SVM_REGRESSOR_GS.csv"):
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_gamma = []
    array_C = []
    array_epsilon = []
    array_means = []
    # param = kwargs
    # print(param)
    # clf = GridSearchCV(gs_svr, param)
    print('\n\n\n\n')

    gs = RandomizedSearchCV(MultiOutputRegressor(svr), param_grid, n_iter= 10, cv = 5)
    gs_svr = gs.fit(x_train,y_train)
    grid_fitted = gs_svr
    # grid_fitted = clf.fit(x, y)
    means = grid_fitted.cv_results_['mean_test_score']
    # means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        array_gamma = array_gamma + [param['estimator__gamma']]
        array_epsilon = array_epsilon + [param['estimator__epsilon']]
        array_C = array_C +  [param ['estimator__C']]
        array_means = array_means +  [mean]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [ array_C , array_gamma , array_epsilon,  array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    df=pd.DataFrame(array_tot)
    df = df.rename(index=str, columns={0: "Parameter_C", 1: "Epsilon" , 2: "Gamma" ,3: "Mean_Val_Error" })
    df.to_csv("./result/SVM_REGRESSOR_GS.csv")
    gh = pd.read_csv("./result/SVM_REGRESSOR_GS.csv")
    print(gh)


def bayesian_func_generator_classification(estimator_func,x,y,n_splits=5):
	def score_func (param_dist):
		estimator = estimator_func(**param_dist)
		score = cross_val_score(estimator, x, y,cv=n_splits).mean()
		return {'loss': (-score), 'status': STATUS_OK}
	return score_func

def hp_tuning_BO(estimator_func,x,y,param_dist,iterations=10,save=True,filename='SVM_CLASS_BO.csv'):
	y=np.ravel(y,order='C')
	objective_function = bayesian_func_generator_classification(estimator_func,x,y)
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
	if save:
			dict_csv={}
			dict_csv.update({'Score' : []})
			for key in vals[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(vals):
				for key in val:
					dict_csv[key].append((val[key])[0])
				dict_csv['Score'].append(losses[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/"+filename),sep=',', index_label='Index')
	return trials


param_dist_RS = {'kernel': ['linear', 'rbf', 'poly'],
              'C': stats.loguniform(1e-4, 1e0)}

param_dist_BO = {
    'C': hp.loguniform('C', np.log(1e-4), np.log(1e2)),
    #'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
    #'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
}

#svc = svm.SVC()
#print("\n\nYOOOOOH\n\n")
#hp_tuning_BO(svm.SVC,x_train,y_train,param_dist_BO,iterations=50)
#hp_tuning_RS(svc,x_train,y_train,param_dist_RS)

########### MI DA PROBLEMI DI IDENTAZIONE NON SO PERCHÈ, VOGLIO MORIRE AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
def hp_tuning_rf_regr_GS(svr, x_train, y_train, param_grid, folds=5, save=True, filename="RF_REGRESSOR_GS.csv"):
    start_time=time.time()

    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_n_estimators = []
    array_max_features = []
    array_max_depth = []
    array_min_samples_split = []
    array_min_samples_leaf = []
    # param = kwargs
    # print(param)
    # clf = GridSearchCV(gs_svr, param)
    print('\n\n\n\n')

    gs = GridSearchCV(MultiOutputRegressor(svr), param_grid=param_grid)
    gs_svr = gs.fit(x_train,y_train)
    grid_fitted = gs_svr
    # grid_fitted = clf.fit(x, y)
    means = grid_fitted.cv_results_['mean_test_score']
    # means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):

        print("%f (%f) with: %r" % (mean, stdev, param))
        # array_n_estimators = array_n_estimators + [param['n_estimators']]
        # array_max_features = array_max_features + [param['max_features'] ]
        # array_max_depth = array_max_depth +  [param ['max_depth']]
        array_min_samples_split = array_min_samples_split + [param ['min_samples_split']]
		# array_min_samples_leaf = array_min_samples_leaf +  [param ['min_samples_leaf']]
        array_means = array_means + [mean]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [ array_max_features , array_min_samples_split,  array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    df=pd.DataFrame(array_tot)
    df = df.rename(index=str, columns={0: "Max_features", 1: "Min_Samples", 2: "Mean_Val_Error" })
    df.to_csv("./result/RF_REGRESSOR_GS.csv")
    gh = pd.read_csv("./result/RF_REGRESSOR_GS.csv")
    print(gh)


def evaluate (x_test, y_test , estimator):
	y_pred = estimator.predict(x_test)
	mean_euclidean_error = 0
	for y_pred, y_test in zip(y_pred, y_test):
		mean_euclidean_error += norm(y_pred - y_test)
	mean_euclidean_error = mean_euclidean_error/len(y_val)
	#
	# ##creo dataset
	# # def main():
	# # 	train_array = gen_dataset.CSV_to_array('./Monk_dataset/monks-1.train', shuf = True, delimiter = ' ')
	# # 	test_array = gen_dataset.CSV_to_array('./Monk_dataset/monks-1.test', shuf = True, delimiter = ' ')
	# # 	x_train = train_array[:,1:7]
	# # 	y_train = train_array[:,0]
	# # 	y_train = y_train.reshape(y_train.shape[0],1)
	# # 	x_test = test_array[:,1:7]
	# # 	y_test = test_array[:,0]
	# # 	y_test = y_test.reshape(y_test.shape[0],1)
	# #
	# # 	x_train = np.asarray(x_train,dtype=np.float64)
	# # 	y_train = np.asarray(y_train,dtype=np.float64)
	# #
	# # 	print (y_train.shape)
	#
	# if __name__ == '__main__':
	#     main()
