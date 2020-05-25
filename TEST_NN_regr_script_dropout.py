import keras
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.utils import shuffle
from keras.constraints import max_norm
from keras.models import model_from_json
from NN_REGR_dropout import NeuralNetwork
import gen_dataset
from hyperopt import hp, pyll
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import os
import json
import matplotlib.pyplot as plt

def save_NN_model(trained_model,filename="prova"):
	# model to JSON
	network_json = trained_model.to_json()
	with open("NN_model_storage/"+filename+"_model.json", "w") as json_file:
		json_file.write(network_json)
	# weights to HDF5
	trained_model.save_weights("NN_model_storage/"+filename+".h5")
	print("Model saved, check folder NN_model_storage/")

def load_NN_model(filename):
	# load json and create model
	json_file = open("NN_model_storage/"+filename+"_model.json", 'r')
	loaded_network_json = json_file.read()
	json_file.close()
	loaded_network = model_from_json(loaded_network_json)
	# load weights into new model
	loaded_network.load_weights("NN_model_storage/"+filename+".h5")
	print("Loaded model from disk")
	return loaded_network

def save_NN_history_dict(history,params_dict,filename):
	history_dict = history.history
	history_dict.update(params_dict)
	print(type(history_dict))
	#history_dict=dict(history_dict)
	json.dump(history_dict, open("NN_model_storage/"+filename+"_history.json", 'w'))

def load_NN_history_dict(filename):
	history_dict = json.load(open("NN_model_storage/"+filename+"_history.json", 'r'))
	return history_dict

# LOAD DATASET
train_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TRAIN.csv', shuf=True, delimiter=',')
test_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TEST.csv', shuf=True, delimiter=',')
x_train,y_train = gen_dataset.divide(train_dataset)
x_test,y_test = gen_dataset.divide(test_dataset)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# GENERATE AND TRAIN MODEL
model_name="5L768_lr0016_mom92_mb209_K5_bis"
architecture = [768, 768, 768, 768, 768]
trial_network = NeuralNetwork(architecture, x_train.shape[1], y_train.shape[1], dropout_rate_hidden=0.2)
loaded_history_dict = load_NN_history_dict(model_name)
params = {
	'lr':loaded_history_dict['lr'], 
	'mom':loaded_history_dict['mom'],
	'decay_rate':loaded_history_dict['decay_rate'],
	'nesterov':loaded_history_dict['nesterov'],
	'epochs':int(loaded_history_dict['Training epochs']),
	'batch_size':loaded_history_dict['batch_size'],#1364,#494,#245
	'activation':loaded_history_dict['activation']}
print(params)
mymodel = trial_network.new_model(**params, loss_function=keras.losses.mean_squared_error)
history, mee = trial_network.train_validate(x_train, y_train, x_test, y_test, patience=3000)
model_name="Test/5L768_lr0016_mom92_mb209_K5_bis_TEST"
save_NN_model(trial_network.last_model_compiled,filename=model_name)
save_NN_history_dict(history, params_dict=params, filename=model_name)
# history
plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_loss'], 'r--')
plt.title('Loss-Epochs\nNetwork Architecture '+str(architecture))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.ylim(0,4)
plt.show()

loaded_network = load_NN_model(filename=model_name)
loaded_network.compile(
	optimizer=keras.optimizers.SGD(
		learning_rate=0.003, 
		momentum=0.85,
		decay=0.0001,
		nesterov=False),
    loss=keras.losses.mean_squared_error)
y_pred = loaded_network.predict(x_test)
mean_euclidean_error = 0
for pred_val, actual_val in zip(y_pred, y_test):
	mean_euclidean_error += np.linalg.norm(pred_val - actual_val)
mean_euclidean_error = mean_euclidean_error/(y_test.shape[0])
print('MEE (test): %.4f' % mean_euclidean_error)
score = loaded_network.evaluate(x_test, y_test, verbose=0)
print("%s: %.4f" % ("MSE (test): ", score))
y_pred_train = loaded_network.predict(x_train)
mean_euclidean_error_train = 0
for pred_val, actual_val in zip(y_pred_train, y_train):
	mean_euclidean_error_train += np.linalg.norm(pred_val - actual_val)
mean_euclidean_error_train = mean_euclidean_error_train/(y_train.shape[0])
print('MEE (train): %.4f' % mean_euclidean_error_train)
score_train = loaded_network.evaluate(x_train, y_train, verbose=0)
print("%s: %.4f" % ("MSE (train): ", score_train))


# history
loaded_history_dict = load_NN_history_dict(model_name)
print("Available keys: ", loaded_history_dict.keys())
plt.plot(loaded_history_dict['loss'], 'b')
plt.plot(loaded_history_dict['val_loss'], 'r--')
#plt.plot(loaded_history_dict['val_loss_std'], 'r:')
plt.title('Loss-Epochs\nNetwork Architecture '+str(architecture))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylim(0,4)
print(loaded_history_dict['loss'][-1])
plt.show()
print(loaded_history_dict['val_loss'][-1])

