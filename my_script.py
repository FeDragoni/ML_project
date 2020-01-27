import keras
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.utils import shuffle
from keras.constraints import max_norm
from NN import NeuralNetwork
import gen_dataset
from hyperopt import hp
import scipy.stats as stats

# def CSV_to_array(file_name, shuf=False, delimiter=',', comment='#'):
#     dataframe = pd.read_csv(file_name, delimiter = delimiter, comment=comment)
#     if shuf :
#         dataframe = shuffle(dataframe)
#         dataframe = dataframe.reset_index(drop=True)
#     array = dataframe.values[:,1:]
#     return array
#print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",np.random.RandomState(1))
train_array = gen_dataset.CSV_to_array(
	'./Monk_dataset/monks-1.train', shuf=True, delimiter=' ')
test_array = gen_dataset.CSV_to_array(
	'./Monk_dataset/monks-1.test', shuf=True, delimiter=' ')
x_train = train_array[:, 1:7]
y_train = train_array[:, 0]
y_train = y_train.reshape(y_train.shape[0], 1)
x_test = test_array[:, 1:7]
y_test = test_array[:, 0]
y_test = y_test.reshape(y_test.shape[0], 1)
trial_network = NeuralNetwork(
	[30, 30, 30], x_train.shape[1], y_train.shape[1], 1, 64,dropout_rate_hidden=0.5)
trial_network.new_model(lr=0.1, mom=0.05, nesterov=True,
						loss_function=keras.losses.binary_crossentropy)
#history, mee = trial_network.train_validate(x_train, y_train, x_test, y_test)

# print(np.mean(history.history['val_loss']))
param_grid = {'epochs':[10], 'batch_size':[32], 'lr':[0.1, 1], 'mom':[0.5]}
param_dist={'epochs':[10],'batch_size':[32],'lr':[0.5]}
param_space = {'lr':hp.loguniform('lr', np.log(0.01), np.log(10)), 'mom':hp.uniform('mom', 0.0, 1.0)}
grid = trial_network.hp_tuning_GS(x_train, y_train, param_grid,folds=2)
#grid = trial_network.hp_tuning_RS(x_train, y_train, param_dist,folds=2,iterations=3)
#trials = trial_network.hp_tuning_BO(x_train, y_train, param_space, iterations=3)
