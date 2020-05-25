import keras
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.constraints import max_norm
from NN import NeuralNetwork
import gen_dataset
from hyperopt import hp, pyll
import scipy.stats as stats

### DATA ACQUISITION
def encode_MONK(x):
	ohe = OneHotEncoder()
	x = ohe.fit_transform(x).toarray()
	return x

train_dataset = pd.read_csv('./Monk_dataset/monks-1.train',delimiter=' ')
x_train = train_dataset.iloc[:, 2:8].values
y_train = train_dataset.iloc[:, 1].values
y_train = y_train.reshape(y_train.shape[0], 1)
test_dataset = pd.read_csv('./Monk_dataset/monks-1.test',delimiter=' ')
x_test = test_dataset.iloc[:, 2:8].values
y_test = test_dataset.iloc[:, 1].values
y_test = y_test.reshape(y_test.shape[0], 1)
x_train = encode_MONK(x_train)
x_test = encode_MONK(x_test)
print(y_train.shape)
print(x_train.shape)

### NN set up
trial_network = NeuralNetwork([17, 17, 17], x_train.shape[1], y_train.shape[1])
trial_network.new_model(lr=0.2, mom=0.6,batch_size=123,
	nesterov=True,activation='relu', loss_function=keras.losses.mean_squared_error,epochs=500,decay_rate=0.0)

### TRAINING
history, mee = trial_network.train_validate(x_train, y_train, x_test, y_test)
print(history.history.keys())
trial_network.show_result(history)
print(np.mean(history.history['val_loss']))

### OPTIMIZATION
param_grid = {'epochs':[5], 'batch_size':[64], 'lr':[0.8], 'mom':[0.8,0.85],'loss_function':[keras.losses.binary_crossentropy],'decay_rate':[0.0, 0.01]}
#param_grid = {'epochs':[500], 'batch_size':[32], 'lr':[0.1], 'mom':[0.4],'loss_function':[keras.losses.binary_crossentropy],'decay_rate':[0.01]}
#param_grid = {'epochs':[50], 'batch_size':[32,64], 'lr':[0.1], 'mom':[0.0]}
param_dist={'epochs':[50,500,1000],'batch_size':[32,64],'lr':[0.5],'loss_function':[keras.losses.binary_crossentropy]}

param_space = {
	#'epochs':pyll.scope.int(hp.loguniform('epochs', np.log(500), np.log(1000))),
	'epochs':hp.choice('epochs', [250]),
	'batch_size':hp.choice('batch_size', [62]),
	'lr':hp.loguniform('lr', np.log(0.01), np.log(1)),
	'mom':hp.uniform('mom', 0.01, 1),
	'decay_rate':hp.loguniform('decay_rate', np.log(0.00001), np.log(0.1)),
	#'decay_rate':hp.choice('decay_rate', [0.0]),
	'loss_function':hp.choice('loss_function', [keras.losses.binary_crossentropy]),
	'activation':hp.choice('activation', ['tanh']),
	'nesterov':hp.choice('nesterov', [False,True])
	}

#grid = trial_network.hp_tuning_GS(x_train, y_train, param_grid,folds=5,filename='prova_GS.csv')
#grid = trial_network.hp_tuning_RS(x_train, y_train, param_dist,folds=2,iterations=3)
#trials = trial_network.hp_tuning_BO(x_train, y_train, param_space, iterations=20,filename='M3_BO_5fold_run2.csv')
