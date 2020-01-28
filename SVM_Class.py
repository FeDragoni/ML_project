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
# import time
import Functions
import scipy.stats as stats
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll


##SVM Classifier

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
x_test = np.asarray(x_test,dtype=np.float64)
y_test = np.asarray(y_test,dtype=np.float64)



print (y_train.shape)

###kernel = rbf
parameters_rbf = {'gamma':[0.01, 0.1, 1, 10, 100, 1000], 'C': [0.01, 0.1, 1, 10, 100, 1000] }
# np.linspace(0.01,1000,7)
parameters_rbf_2 = {'gamma':[x for x in np.linspace(start = 0.01, stop = 1, num = 1000)],
                    'C':[x for x in np.linspace(start = 20, stop = 1000, num = 1000)]}


###kernel = poly
parameters_poly = {'gamma':[0.01, 0.1, 1, 10, 100, 1000], 'C': [0.01, 0.1, 1, 10, 100, 1000] , 'degree' : [1,2,3], 'coef0' : [1,2,3] }

param_dist_BO = {
    # 'gamma': hp.quniform('gamma', 0.001, 10 , 1),
    'gamma': (hp.quniform('gamma', 0.01, 0.07, 0.001)),
    'C': hp.quniform('C', 280, 340, 1)
    #'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
}


svc = svm.SVC()


# Functions.hp_tuning_GS( svc, x_train, y_train,parameters_rbf_2, folds=5, save=True, filename="SVM_CLASS_GS_rbf_2.csv",)
# Functions.hp_tuning_svm_RS( svc, x_train, y_train,parameters_rbf, folds=5, save=True, filename="SVM_CLASS_RS.csv",)
Functions.hp_tuning_BO( svm.SVC, x_train, y_train, param_dist_BO , save=True, filename="SVM_CLASS_BO.csv",)

# gh = pd.read_csv("./result/1.csv")
# print(gh)











##parametri
# def hp_tuning_svm_GS(svm, x, y, **kwargs):
#     start_time = time.time()
#     y=np.ravel(y,order='C')
#     parameters = kwargs
#     clf = GridSearchCV(svm, parameters)
#     grid_fitted = clf.fit(x, y)
#     print("Time used for grid search: %.3f" %(time.time()-start_time))
#     means = grid_fitted.cv_results_['mean_test_score']
#     stds = grid_fitted.cv_results_['std_test_score']
#     params = grid_fitted.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#         array_kernel = array_kernel + [param['kernel']]
#         array_C = array_C +  [param ['C']]
#         array_means = array_means + [ mean ]
#     print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
#     df=pd.DataFrame(zip(array_kernel,array_C,array_means))
#     df = df.rename(index=str, columns={0: "mean Validation Error", 1: "Parameter_C", 2: "Kernel" })
#     df.to_csv("./result/SVMM_func.csv")
#     print("Total elapsed time: %.3f" %(time.time()-start_time))

# def hp_tuning_svm_GS(svm, x, y, **kwargs)


###inizio mio vecchio codice per scrivere in csv
# clf = GridSearchCV(svc, parameters)
# grid_fitted = clf.fit(x_train, np.ravel(y_train,order='C'))

# array_kernel = []
# array_C = []
# array_means = []
# means = grid_fitted.cv_results_['mean_test_score']
# # means_train = grid_fitted.cv_results_['mean_train_score']
# stds = grid_fitted.cv_results_['std_test_score']
# params = grid_fitted.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#     array_kernel = array_kernel + [param['kernel']]
#     array_C = array_C +  [param ['C']]
#     array_means = array_means + [ mean ]
#
# print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
# print (array_kernel)
# print (array_C)
# print(means)
#
# # array_tot = array_kernel.append(array_C)
# array_tot = [array_means, array_C , array_kernel]
# array_tot = zip(*array_tot)
# print (array_tot)
#
# df = pd.DataFrame(array_tot)
# df = df.rename(index=str, columns={0: "mean Validation Error", 1: "Parameter_C", 2: "Kernel" })
# df.to_csv("./result/SVMM.csv")
#
# gh = pd.read_csv("./result/SVMM.csv")
# print(gh)
# print(np.shape(gh))
####fine codice per scrivere in csv e fare HP tuning


##CAZZO DEVO ANCHE VALUTARE, per l'accuracy


# def from_param_to_model ()
################DA FARE
#####bestmodel = grid_fitted.best_estimator_
#####base_accuracy = evaluate(bestmodel, x_test, y_test)
#####print (base_accuracy)
