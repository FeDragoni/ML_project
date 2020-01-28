import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.constraints import max_norm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import max_norm
from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def CSV_to_array(file_name, shuf=False, delimiter=',', comment='#'):
    dataframe = pd.read_csv(file_name, delimiter = delimiter, comment=comment)
    if shuf :
        dataframe = shuffle(dataframe)
        dataframe = dataframe.reset_index(drop=True)
    array = dataframe.values[:,1:]
    return array
 
# Function to create model, required for KerasClassifier
architecture = [10, 30, 30, 10]
input_dim = 6
output_dim = 1
def create_model(optimizer='rmsprop', init='glorot_uniform', activation='relu', loss='binary_crossentropy'):
	# create model
	model = Sequential()
	model.add(Dense(architecture[0], input_dim=input_dim, kernel_initializer=init, activation=activation))
	for layer_units in architecture :
		model.add(Dense(layer_units, kernel_initializer=init, activation=activation))
	model.add(Dense(output_dim, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model
 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# split into input (X) and output (Y) variables
train_array = CSV_to_array('./Monk_dataset/monks-1.train', shuf = True, delimiter = ' ')
test_array = CSV_to_array('./Monk_dataset/monks-1.test', shuf = True, delimiter = ' ')
x_train = train_array[:,1:7]
y_train = train_array[:,0]
y_train = y_train.reshape(y_train.shape[0],1)
x_test = test_array[:,1:7]
y_test = test_array[:,0]
y_test = y_test.reshape(y_test.shape[0],1)
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
param_grid = dict(optimizer = ['rmsprop', 'adagrad'])
param_grid['init'] = ['glorot_uniform', 'normal', 'uniform']
param_grid['epochs'] = [50, 100, 500]
#param_grid['batch_size'] = [5, 10, 20, 100]
param_grid['loss'] = ['binary_crossentropy', 'hinge']
print('\n\n')
print(param_grid)
print('\n\n')
#param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, )
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))