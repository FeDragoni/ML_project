## random forest Classifier

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

from hyperopt import hp, pyll

# import validation
import csv
from sklearn.model_selection import GridSearchCV
import time
# import ottimizzazioni_RF
import Functions



#################  levare le librerie che non servono a nulla
#creo dataset

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

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform (x_train)

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
parameters = {#'n_estimators': n_estimators,
                   'max_features': max_features,
                    #'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                    #'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
print(parameters)

rf = RandomForestClassifier()
print("ciao")

param_space_BO={
	'n_estimators': pyll.scope.int(hp.loguniform('n_estimators', np.log(10), np.log(10000))),
	'min_samples_split': pyll.scope.int(hp.quniform('min_samples_split', 2, 50, 1)),
	'max_features': hp.choice('max_features', ['auto', 'sqrt'])
}

# Functions.hp_tuning_svm_GS( rf, x_train, y_train,parameters, folds=5, save=True, filename="RF_CLASS_GS.csv")
# Functions.hp_tuning_svm_RS( rf, x_train, y_train,parameters, folds=5, save=True, filename="RF_CLASS_RS.csv")
Functions.hp_tuning_BO( RandomForestClassifier ,x_train ,y_train ,param_space_BO ,iterations=10,save=True,filename='RF_CLASS_BO.csv')



# ottimizzazioni_RF.hp_tuning_GS(x_train, y_train, rf, folds=5, save =True,   filename = "RF_GS.csv", **random_grid)
    ##questo è per valutare il BEST!!!
    # best_grid = grid_search.best_estimator_
    # grid_accuracy = evaluate(best_grid, test_features, test_labels)
