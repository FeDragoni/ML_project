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
import time

from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import RepeatedKFold


def hp_tuning_GS(x, y, estimator, folds=5, save=True, filename="SVM_CLASS_GS.csv", **kwargs):
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_kernel = []
    array_C = []
    array_means = []
    param = kwargs
    print(param)
    clf = GridSearchCV(estimator, param)
    print('\n\n\n\n')
    grid_fitted = clf.fit(x, y)
    means = grid_fitted.cv_results_['mean_test_score']
    # means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        array_kernel = array_kernel + [param['kernel']]
        array_C = array_C +  [param ['C']]
        array_means = array_means +  [mean]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [array_kernel, array_C , array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    if save:
        df=pd.DataFrame(array_tot)
        df = df.rename(index=str, columns={0: "mean Validation Error", 1: "Parameter_C", 2: "Kernel"})
        df.to_csv("./result/"+filename)
