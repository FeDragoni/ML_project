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

# import validation
import csv
from sklearn.model_selection import GridSearchCV
import time
import ottimizzazioni_RF


#################  levare le librerie che non servono a nulla
#creo dataset
def main():
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

    rf = RandomForestClassifier()
    print("ciao")

    ottimizzazioni_RF.hp_tuning_GS(x_train, y_train, rf, folds=5, save =True,   filename = "RF_GS.csv", **random_grid)
    ##questo è per valutare il BEST!!!
    # best_grid = grid_search.best_estimator_
    # grid_accuracy = evaluate(best_grid, test_features, test_labels)

if __name__ == "__main__":
   main()
