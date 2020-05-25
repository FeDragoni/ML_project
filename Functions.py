import numpy as np
import itertools
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import csv
import time
import scipy.stats as stats
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Grid search - MONK classifier
def hp_tuning_GS(estimator, x, y, param_grid, folds=5, save=True, filename="prova_GS.csv"):
    start_time = time.time()
    y=np.ravel(y,order='C')
    clf = GridSearchCV(estimator, param_grid, cv=folds,n_jobs=2)
    grid_fitted = clf.fit(x, y)
    time_used = time.time()-start_time
    means_test = grid_fitted.cv_results_['mean_test_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means_test, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Time used for grid search: " , time_used)
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        dict_csv.update({'Std' : []})
        for key in params[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(params):
            for key in val:
                dict_csv[key].append((val[key]))
            dict_csv['Score'].append(means_test[index])
            dict_csv['Std'].append(stds[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df.to_csv(path_or_buf=("./result/RF_class_ken/post/"+filename),sep=',', index_label='Index')
    return grid_fitted

# Random search - MONK classifier
def hp_tuning_RS(estimator, x, y, param_dist, iterations=10, folds=5, save=True, filename="prova_RS.csv"):
    # NOTE: continuous parameters should be given as a distribution for a proper random search
    # Distributions can be generated with scipy.stats module
    # For parameters that need to be explored in terms of order of magnitude loguniform distribution is recommended
    start_time = time.time()
    y=np.ravel(y,order='C')
    clf = RandomizedSearchCV(estimator, param_dist, n_iter=iterations, cv=folds,n_jobs=2)
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
        dict_csv.update({'Std' : []})
        for key in params[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(params):
            for key in val:
                dict_csv[key].append((val[key]))
            dict_csv['Score'].append(means_test[index])
            dict_csv['Std'].append(stds[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df.to_csv(path_or_buf=("./result/RF_Class/M3/"+filename),sep=',', index_label='Index')
    return grid_fitted

# Grid search SVM - Multioutput classifier
def hp_tuning_svm_regr_GS(estimator, x, y, param_grid, folds=5, save=True, filename="prova_GS.csv"):
    start_time = time.time()
    clf = GridSearchCV(MultiOutputRegressor(estimator, n_jobs= 2), param_grid, cv=folds,n_jobs=2, return_train_score=True)
    print(x.shape)
    print(y.shape)
    grid_fitted = clf.fit(x, y)
    time_used = time.time()-start_time
    means_test = grid_fitted.cv_results_['mean_test_score']
    stds_test = grid_fitted.cv_results_['std_test_score']
    means_train = grid_fitted.cv_results_['mean_train_score']
    stds_train = grid_fitted.cv_results_['std_train_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means_test, stds_test, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Time used for grid search: " , time_used)
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        dict_csv.update({'Std' : []})
        dict_csv.update({'Score TRAIN' : []})
        dict_csv.update({'Std Dev TRAIN' : []})
        for key in params[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(params):
            for key in val:
                dict_csv[key].append((val[key]))
            dict_csv['Score'].append(means_test[index])
            dict_csv['Std'].append(stds_test[index])
            dict_csv['Score TRAIN'].append(means_train[index])
            dict_csv['Std Dev TRAIN'].append(stds_train[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df.to_csv(path_or_buf=("./result/SVM_Regr/"+filename),sep=',', index_label='Index')
    return grid_fitted

# Grid search RF
def hp_tuning_regr_RF_GS(estimator, x, y, param_grid, folds=5, save=True, filename="prova_regr_RF_GS.csv"):
    start_time = time.time()
    clf = GridSearchCV(estimator, param_grid = param_grid, cv=folds,scoring='r2',n_jobs=2, return_train_score=True)
    grid_fitted = clf.fit(x, y)
    time_used = time.time()-start_time
    means_test = grid_fitted.cv_results_['mean_test_score']
    stds_test = grid_fitted.cv_results_['std_test_score']
    means_train = grid_fitted.cv_results_['mean_train_score']
    stds_train = grid_fitted.cv_results_['std_train_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means_test, stds_test, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Time used for grid search: " , time_used)
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    if save:
        dict_csv={}
        dict_csv.update({'Score' : []})
        dict_csv.update({'Std Dev' : []})
        dict_csv.update({'Score TRAIN' : []})
        dict_csv.update({'Std Dev TRAIN' : []})
        for key in params[0]:
            dict_csv.update({key : []})
        for index,val in enumerate(params):
            for key in val:
                dict_csv[key].append((val[key]))
            dict_csv['Score'].append(means_test[index])
            dict_csv['Std Dev'].append(stds_test[index])
            dict_csv['Score TRAIN'].append(means_train[index])
            dict_csv['Std Dev TRAIN'].append(stds_train[index])
        df = pd.DataFrame.from_dict(dict_csv, orient='columns')
        df.to_csv(path_or_buf=("./result/"+filename),sep=',', index_label='Index')
    return grid_fitted

# Bayesian optimization objective_function generator
def bayesian_func_generator_classification(estimator_func,x,y,n_splits=5):
    def score_func (param_dist):
        estimator = estimator_func(**param_dist)
        score = cross_val_score(estimator, x, y,cv=n_splits).mean()
        return {'loss': (-score), 'status': STATUS_OK}
    return score_func
# Bayesian optimization
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
        df.to_csv(path_or_buf=("./result/RF_Class/M3/"+filename),sep=',', index_label='Index')
    return trials

