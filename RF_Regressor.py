import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split,  GridSearchCV, cross_val_score
import gen_dataset
import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import csv
import time
import Functions


train_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TRAIN.csv', shuf=True, delimiter=',')
test_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TEST.csv', shuf=True, delimiter=',')
x_train,y_train=gen_dataset.divide(train_dataset)
x_test,y_test=gen_dataset.divide(test_dataset)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x train: ",x_train.shape)
print("y train: ",y_train.shape)
print("x test: ",x_test.shape)
print("y test: ",y_test.shape)

# Number of trees in random forest
n_estimators = [int(x) for x in np.logspace(start = np.log10(100), stop = np.log10(2000), num = 7)]
# Number of features to consider at every split
max_features = ['auto','sqrt']
#max_features = ['auto','sqrt']
    # Minimum number of samples required to split a node
min_samples_split = [2]
bootstrap = [False]

rf = RandomForestRegressor()
# parameters = {
#     'estimator__n_estimators': n_estimators,
#     'estimator__max_features': max_features,
#     'estimator__min_samples_split': min_samples_split,
#     'estimator__bootstrap': bootstrap}
parameters = {
    'n_estimators': [700],
    'max_features': [3],
    'bootstrap': [False]}

# HP tuning

grid_fitted=Functions.hp_tuning_regr_RF_GS(rf, x_train, y_train,parameters, folds=5, save=True, filename="RF_regr_ken/PROVA_RF_REGR_final.csv")
best_rf = svm.SVR(**grid_fitted.best_params_)

# Print score

myRF = RandomForestRegressor(n_estimators=700,max_features=3,bootstrap=False)
myRF.fit(x_train, y_train)
y_pred = myRF.predict(x_test)
y_pred_train = myRF.predict(x_train)

print('TEST')
print('MSE: ',sklearn.metrics.mean_squared_error(y_test,y_pred))
print('MAE: ',sklearn.metrics.mean_absolute_error(y_test,y_pred))
print('R^2: ',myRF.score (x_test, y_test))
MEE = 0
for pred_val, actual_val in zip(y_pred, y_test):
	MEE += np.linalg.norm(pred_val - actual_val)
MEE = MEE/(y_train.shape[0])
print('MEE: ', MEE)

print('TRAIN')
print('MSE_train: ',sklearn.metrics.mean_squared_error(y_train,y_pred_train))
print('MAE_train: ',sklearn.metrics.mean_absolute_error(y_train,y_pred_train))
print('R^2_train: ',myRF.score (x_train, y_train))
MEE_train = 0
for pred_val, actual_val in zip(y_pred_train, y_train):
	MEE_train += np.linalg.norm(pred_val - actual_val)
MEE_train = MEE_train/(y_train.shape[0])
print('MEE_train: ', MEE_train)

