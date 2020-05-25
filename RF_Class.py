import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
import csv
import gen_dataset
import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, pyll
import Functions
from sklearn.preprocessing import StandardScaler, OneHotEncoder

### DATA ACQUISITION
def encode_MONK(x):
	ohe = OneHotEncoder()
	x = ohe.fit_transform(x).toarray()
	return x

train_dataset = pd.read_csv('./Monk_dataset/monks-2.train',delimiter=' ')
x_train = train_dataset.iloc[:, 2:8].values
y_train = train_dataset.iloc[:, 1].values
y_train = y_train.reshape(y_train.shape[0], 1)

test_dataset = pd.read_csv('./Monk_dataset/monks-2.test',delimiter=' ')
x_test = test_dataset.iloc[:, 2:8].values
y_test = test_dataset.iloc[:, 1].values
y_test = y_test.reshape(y_test.shape[0], 1)

x_train = encode_MONK(x_train)
x_test = encode_MONK(x_test)

print(y_train.shape)
print(x_train.shape)
x_train = np.asarray(x_train,dtype=np.float64)
y_train = np.asarray(y_train,dtype=np.float64)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = [9,10,11,12,13,14,15]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 60, num = 6)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2, 10, num = 6)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 15, num = 6)]
# Method of selecting samples for training each tree
bootstrap = [False]# Create the random grid
parameters = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'bootstrap' : bootstrap ,
    'max_depth': max_depth,
    'min_samples_split': [1],
    'min_samples_leaf': [1],
}
print(parameters)

rf = RandomForestClassifier()

grid_fitted = Functions.hp_tuning_GS(rf, x_train, y_train,parameters, folds=5, save=True, filename="M2_RF_GS_run3_minleaf1.csv")
best_one = RandomForestClassifier(**grid_fitted.best_params_)

# best_one = RandomForestClassifier(
#     n_estimators = 100,
#     max_features=9,
#     bootstrap = False,
#     min_samples_leaf = 1)
best_one.fit (x_train, np.ravel(y_train,order='C'))

print ("Train accuracy: ",best_one.score (x_train, y_train))
print ("Test accuracy: ",best_one.score (x_test, y_test))
