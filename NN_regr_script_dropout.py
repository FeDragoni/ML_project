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

def save_NN_history_dict(history_dict,params_dict,filename):
	history_dict.update(params_dict)
	print(type(history_dict))
	#history_dict=dict(history_dict)
	json.dump(history_dict, open("NN_model_storage/"+filename+"_history.json", 'w'))

def load_NN_history_dict(filename):
	history_dict = json.load(open("NN_model_storage/"+filename+"_history.json", 'r'))
	return history_dict


# LOAD DATASET
validation_split=False #set to true if not using K-fold evaluation
train_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TRAIN.csv', shuf=True, delimiter=',')
test_dataset = gen_dataset.CSV_to_array('./dataset/OUR_TEST.csv', shuf=True, delimiter=',')
scaler = StandardScaler()
if validation_split:
	x_train,y_train = gen_dataset.divide(train_dataset[:-200])
	x_val,y_val = gen_dataset.divide(train_dataset[-200:])
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_val = scaler.transform(x_val)
else :
	x_train,y_train = gen_dataset.divide(train_dataset)
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
x_test,y_test = gen_dataset.divide(test_dataset)
x_test = scaler.transform(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# GENERATE AND TRAIN MODEL
model_name="5L768_lr0016_mom92_mb209_K5_bis"
architecture = [768, 768, 768, 768, 768]
trial_network = NeuralNetwork(architecture, x_train.shape[1], y_train.shape[1], dropout_rate_hidden=0.2)
params = {
	'lr':0.0016, 
	'mom':0.92,
	'decay_rate':0.0,
	'nesterov':False,
	'epochs':3000,
	'batch_size':209,#1564,#524,#261
	'activation':'relu'}
mymodel = trial_network.new_model(**params, loss_function=keras.losses.mean_squared_error)
if validation_split:
	history, mee = trial_network.train_validate(x_train, y_train, x_val, y_val, patience=400)
	history_dict = history.history
else:
	kfold_histories, kfold_mee = trial_network.k_fold(
		x_train,y_train,
		n_splits=5,
		show_single=False,
		patience=3000)
	#values are plotted only to epochs = min_epochs
	min_epochs = np.min([x.history["Training epochs"] for x in kfold_histories])
	val_loss_store = np.zeros((len(kfold_histories),min_epochs))
	loss_store = np.zeros((len(kfold_histories),min_epochs))
	training_time_store = np.zeros((len(kfold_histories),))
	training_epochs_store = np.zeros((len(kfold_histories),))

	for i,hist in enumerate(kfold_histories):
		val_loss_store[i] = hist.history['val_loss'][:min_epochs]
		loss_store[i] = hist.history['loss'][:min_epochs]
		training_time_store[i] = hist.history["Training time"]
		training_epochs_store[i] = hist.history["Training epochs"]
	val_loss_mean = np.ndarray.tolist(np.mean(val_loss_store,axis=0))
	val_loss_std = np.ndarray.tolist(np.std(val_loss_store,axis=0))
	loss_mean = np.ndarray.tolist(np.mean(loss_store,axis=0))
	loss_std = np.ndarray.tolist(np.std(loss_store,axis=0))
	training_time_mean = np.mean(training_time_store,axis=0)
	training_time_std = np.std(training_time_store,axis=0)
	training_epochs_mean = np.mean(training_epochs_store)
	training_epochs_std = np.std(training_epochs_store)
	history_dict = {
		'val_loss':val_loss_mean, 'val_loss_std':val_loss_std,
		'loss':loss_mean, 'loss_std':loss_std,
		'Training time':training_time_mean, 'Training time std':training_time_std,
		'Training epochs':training_epochs_mean, 'Training epochs std':training_epochs_std,}
	print("Mean epochs", history_dict['Training epochs'], "with std: ", history_dict['Training epochs std'])

# SAVE MODEL AND TRAINING
save_NN_model(trial_network.last_model_compiled,filename=model_name)
#save_NN_history_dict(history.history, params_dict=params, filename=model_name)
save_NN_history_dict(history_dict, params_dict=params, filename=model_name)

# LOAD MODEL AND SHOW RESULTS
# result
loaded_network = load_NN_model(filename=model_name)
loaded_network.compile(
	optimizer=keras.optimizers.SGD(
		learning_rate=0.0017, 
		momentum=0.92,
		decay=0.0,
		nesterov=False),
    loss=keras.losses.mean_squared_error)
y_pred = loaded_network.predict(x_test)
mean_euclidean_error = 0
for pred_val, actual_val in zip(y_pred, y_test):
	mean_euclidean_error += np.linalg.norm(pred_val - actual_val)
mean_euclidean_error = mean_euclidean_error/(y_test.shape[0])
print('MEE: %.4f' % mean_euclidean_error)
score = loaded_network.evaluate(x_test, y_test, verbose=0)
print("%s: %.4f" % ("MSE: ", score))
# history
loaded_history_dict = load_NN_history_dict(model_name)
print("Available keys: ", loaded_history_dict.keys())
plt.plot(loaded_history_dict['loss'], 'b')
plt.plot(loaded_history_dict['val_loss'], 'r--')
plt.plot(loaded_history_dict['val_loss_std'], 'r:')
plt.title('Loss-Epochs\nNetwork Architecture '+str(architecture))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Validation Std.'], loc='upper right')
plt.ylim(0,4)
plt.show()
print("Loss Std: ",loaded_history_dict['loss_std'][-1])
print("Val Loss Std: ",loaded_history_dict['val_loss_std'][-1])
print("Mean Loss: ", loaded_history_dict['loss'][-1])
print("Mean Val Loss: ", loaded_history_dict['val_loss'][-1])
print("Training time: ", loaded_history_dict['Training time'], " for ", int(loaded_history_dict['Training epochs']), " epochs")
