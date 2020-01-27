import numpy as np
import itertools
import pandas as pd
# import datacontrol
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
import gen_dataset
# import validation
from sklearn.svm import SVR
import csv
from sklearn.model_selection import GridSearchCV
# import time
import SVM_utils
import Functions


dati_tot = gen_dataset.readFile("./dataset/OUR_TRAIN.csv")
x_train, y_train = gen_dataset.divide(dati_tot)
print (np.shape(y_train))


param_grid_RBF = {'estimator__C':[0.01] ,
               'estimator__gamma':[0.01] ,
                'estimator__epsilon' : [0.1, 0.01]
              }

param_grid_POLY = {'estimator__C':[0.01] ,
               'estimator__gamma':[0.01] ,
                'estimator__epsilon' : [0.1, 0.01],
                'estimator__degree' : [1,2,3,4],
              }
###CON KERNEL=POLY BISOGNA AGGIUNGERE UN ARRAY NEI VARI OTTTIMIZZATORI DI HP ........ CAZZOO
gb = svm.SVR()

# Functions.hp_tuning_svm_regr_GS(gb, x_train, y_train,param_grid )
# Functions.hp_tuning_svm_regr_RS(gb, x_train, y_train,param_grid )
Functions.hp_tuning_svm_BO(gb, x_train, y_train,param_grid )








# gs = GridSearchCV(MultiOutputRegressor(gb), param_grid=param_grid)
# gs_svr = gs.fit(x_train,y_train)
# grid_fitted = gs_svr
# print (gs_svr.best_estimator)


# def hp_tuning_svmR_GS(svr, x_train, y_train, param_grid, folds=5, save=True, filename="SVM_REGRESSOR_GS.csv"):
#
# ####TUNING GridSearchCV
#     start_time=time.time()
#     # Parameters to be optimized can be choosen between the parameters of self.new_model and are
#     # given through **kwargs as --> parameter=[list of values to try for tuning]
#     # NOTE: batch_size and epochs can also be choosen
#     #The CSV file with the result is saved inside the result/ folder
#     print("ciao")
#     array_gamma = []
#     array_C = []
#     array_epsilon = []
#     array_means = []
#     # param = kwargs
#     # print(param)
#     # clf = GridSearchCV(gs_svr, param)
#     print('\n\n\n\n')
#
#     gs = GridSearchCV(MultiOutputRegressor(svr), param_grid=param_grid)
#     gs_svr = gs.fit(x_train,y_train)
#     grid_fitted = gs_svr
#     # grid_fitted = clf.fit(x, y)
#     means = grid_fitted.cv_results_['mean_test_score']
#     # means_train = grid_fitted.cv_results_['mean_train_score']
#     stds = grid_fitted.cv_results_['std_test_score']
#     params = grid_fitted.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#         array_gamma = array_gamma + [param['estimator__gamma']]
#         array_epsilon = array_epsilon + [param['estimator__epsilon']]
#         array_C = array_C +  [param ['estimator__C']]
#         array_means = array_means +  [mean]
#     print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
#     # array_tot = array_kernel.append(array_C)
#     array_tot = [ array_C , array_gamma , array_epsilon,  array_means]
#     array_tot = zip(*array_tot)
#     print (array_tot)
#     print('Total elapsed time: %.3f' %(time.time()-start_time))
#     df=pd.DataFrame(array_tot)
#     df = df.rename(index=str, columns={0: "Parameter_C", 1: "Epsilon" , 2: "Gamma" ,3: "Mean_Val_Error" })
#     df.to_csv("./result/SVM_REGRESSOR_GS.csv")
#     gh = pd.read_csv("./result/SVM_REGRESSOR_GS.csv")
#     print(gh)


###ALTRO MODO

# pipe_svr = Pipeline([('scl', StandardScaler()),
#         ('reg', MultiOutputRegressor(SVR()))])
# grid_param_svr = {
#     'reg__estimator__C': [0.1,1,10]
# }
#
# gs_svr = (GridSearchCV(estimator=pipe_svr,
#                       param_grid=grid_param_svr,
#                       cv=2,
#                       scoring = 'neg_mean_squared_error',
#                       n_jobs = -1))
#
# gs_svr = gs_svr.fit(x_train,y_train)
# print (gs_svr.best_estimator_)

############################################################

#########################################################à

# parameters = {'reg__estimator__kernel':('linear', 'rbf'), 'reg__estimator__C':[1, 10]}
# svr = svm.SVR()
# SVRegressor = MultiOutputRegressor(svr, n_jobs=2)
# # print (SVRegressor.get_params().keys())
# SVM_utils.hp_tuning_GS(x_train, y_train, SVRegressor, folds=5, save=True, filename="SVM_REGRESSOR_GS.csv", **parameters)
