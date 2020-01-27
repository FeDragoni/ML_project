## random forest REGRESSOR!!!!!!!!!!!!

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
# import matplotlib.pyplot as plt
# import time
# import validation
import csv
import gen_dataset
# import keras

import sklearn.preprocessing
from sklearn.utils import shuffle
# from keras.constraints import max_norm
# from pprint import pprint
# from sklearn import metrics
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
import ottimizzazioni_RF


#################  levare le librerie che non servono a nulla
#creo dataset
def main():
##creo dataset
    dati_tot = gen_dataset.readFile("./dataset/OUR_TRAIN.csv")
    x_train, y_train = gen_dataset.divide(dati_tot)


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 2)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 2)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {#'n_estimators': n_estimators,
                   'max_features': max_features,
                    #'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                    #'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)

    rf = RandomForestRegressor()
    param_grid = {'estimator__C':[10, 50]}
                  #'estimator__kernel':('linear', 'rbf')}

    rf = GridSearchCV(MultiOutputRegressor(rf), param_grid=param_grid)
    gs_svr = gs.fit(x_train,y_train)
    grid_fitted = gs_svr
    print("ciao")

    ####TUNING GridSearchCV
    start_time=time.time()
    # Parameters to be optimized can be choosen between the parameters of self.new_model and are
    # given through **kwargs as --> parameter=[list of values to try for tuning]
    # NOTE: batch_size and epochs can also be choosen
    #The CSV file with the result is saved inside the result/ folder
    print("ciao")
    array_kernel = []
    array_C = []
    array_means = []
    # param = kwargs
    # print(param)
    # clf = GridSearchCV(gs_svr, param)
    print('\n\n\n\n')
    # grid_fitted = clf.fit(x, y)
    means = grid_fitted.cv_results_['mean_test_score']
    # means_train = grid_fitted.cv_results_['mean_train_score']
    stds = grid_fitted.cv_results_['std_test_score']
    params = grid_fitted.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        # array_kernel = array_kernel + [param['kernel']]
        array_C = array_C +  [param ['estimator__C']]
        array_means = array_means +  [mean]
    print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
    # array_tot = array_kernel.append(array_C)
    array_tot = [ array_C , array_means]
    array_tot = zip(*array_tot)
    print (array_tot)
    print('Total elapsed time: %.3f' %(time.time()-start_time))
    df=pd.DataFrame(array_tot)
    df = df.rename(index=str, columns={0: "mean Validation Error", 1: "Parameter_C", 2: "Kernel"})
    df.to_csv("./result/SVM_REGRESSOR_GS.csv")


    # ottimizzazioni_RF.hp_tuning_GS(x_train, y_train, rf, folds=5, save =True,   filename = "RF_GS.csv", **random_grid)


if __name__ == "__main__":
   main()
