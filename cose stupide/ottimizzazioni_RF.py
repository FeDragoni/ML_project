
# import create_dataset
import numpy as np
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
# import validation
import csv
import gen_dataset
# import keras

import sklearn.preprocessing
from sklearn.utils import shuffle
# from keras.constraints import max_norm
from pprint import pprint
from sklearn import metrics
####TRY RANDOMIZED SEARCH CV FOR HP
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# import validation
import csv
from sklearn.model_selection import GridSearchCV
import time




def hp_tuning_GS(x, y, estimator, folds=5, save=True, filename="NN_GS.csv", **kwargs):
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_features = []
    array_samples_split = []
    array_means = []
    param = kwargs
    print(param)
    clf = GridSearchCV(estimator, param)
    print('\n\n\n\n')
    grid_fitted = clf.fit(x, np.ravel(y,order='C'))
    means = grid_fitted.cv_results_['mean_test_score']
    means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        array_features = array_features + [param['max_features']]
        array_samples_split = array_samples_split +  [param ['min_samples_split']]
        array_means = array_means + [param ['bootstrap']]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [array_features, array_samples_split , array_means]
    array_tot = zip(*array_tot)

    value_on_test =
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    if save:
        df=pd.DataFrame(array_tot)
        df = df.rename(index=str, columns={0: "mean Validation Score",1:"mean Train Score",2: "Parameters"})
        df.to_csv("./result/"+filename)



def hp_tuning_RS(x, y, estimator, folds=5, save=True, filename="NN_GS.csv", **kwargs):
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_features = []
    array_samples_split = []
    array_means = []
    param = kwargs
    print(param)
    clf  = RandomizedSearchCV(estimator = rf, param_distributions = param, n_iter = 100,
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)
    print('\n\n\n\n')
    grid_fitted = clf.fit(x, np.ravel(y,order='C'))
    means = grid_fitted.cv_results_['mean_test_score']
    # means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        array_features = array_features + [param['max_features']]
        array_samples_split = array_samples_split +  [param ['min_samples_split']]
        array_means = array_means + [param ['bootstrap']]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [array_features, array_samples_split , array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    if save:
        df=pd.DataFrame(array_tot)
        df = df.rename(index=str, columns={0: "mean Validation Score",1:"mean Train Score",2: "Parameters"})
        df.to_csv("./result/"+filename)
