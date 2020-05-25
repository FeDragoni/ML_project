from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.constraints import max_norm
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class NeuralNetwork():
	def __init__(self, architecture, input_dimension, output_dimension, classification=True, dropout_rate_input=0, dropout_rate_hidden=0):
		self.architecture = architecture  # number of units in each layer
		self.epochs = 0
		self.batch_size = 0
		self.dropout_rate_input = dropout_rate_input
		self.dropout_rate_hidden = dropout_rate_hidden
		self.last_model_compiled = Sequential()
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.classification = classification

	def new_model(self, activation='sigmoid', kernel_initializer='normal',
				kernel_constraint=4.0, loss_function=keras.losses.binary_crossentropy,
				lr=0.01, mom=0.0, nesterov=False,decay_rate=0.00, epochs=50, batch_size=32):
		self.epochs=epochs
		self.batch_size=batch_size
		model = Sequential()
		# add input layer
		if self.dropout_rate_input > 0:
			model.add(Dropout(self.dropout_rate_input))
		model.add(Dense(kernel_initializer=kernel_initializer, kernel_constraint=max_norm(kernel_constraint),
					units=self.architecture[0], activation=activation, input_dim=self.input_dimension))
		# add hidden nodes, w/ dropout if given
		for layer in self.architecture[1:]:
			if self.dropout_rate_hidden > 0:
				model.add(Dropout(self.dropout_rate_hidden))
			model.add(Dense(kernel_initializer=kernel_initializer, kernel_constraint=max_norm(kernel_constraint),
					units=layer, activation=activation))
        # output layer and model compilation NOTE: activation function of last layer need some considerations (e.g. using
        # a linear activation function for binary classification may obviously not be the best choice)
		model.add(Dense(units=self.output_dimension, activation='sigmoid'))
		# NOTE: maybe it would be better to choose 'binary_accuracy' to evaluate model performance for classification
		model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nesterov,decay=decay_rate),
                        loss=loss_function, metrics=['binary_accuracy'])
		self.last_model_compiled = model
		return model

	def train_validate(self, x_train, y_train, x_val, y_val, last_model=True, model=None):
		start_time = time.time()
		if last_model:
			model = self.last_model_compiled
		else:
			model = model
		# scaler = StandardScaler()
		# x_train = scaler.fit_transform(x_train)
		# x_val = scaler.transform(x_val)
		history = model.fit(x=x_train, y=y_train, epochs=self.epochs,
                            shuffle=True, batch_size=self.batch_size,
                            validation_data=(x_val, y_val), verbose=2)
		y_pred = model.predict(x_val)
		mean_euclidean_error = 0
		for pred_val, actual_val in zip(y_pred, y_val):
			mean_euclidean_error += norm(pred_val - actual_val)
		mean_euclidean_error = mean_euclidean_error/len(y_val)
		print('\ntime: %.3f s' % (time.time()-start_time))
		print('Mean Euclidean error: %.3f' % mean_euclidean_error)
		print('y_pred (%f) and y_val (%f) length' % (len(y_pred), len(y_val)))
		return history, mean_euclidean_error

	def show_result(self, history):  # should we also print accuracy?
		plt.plot(history.history['binary_accuracy'], 'b')
		plt.plot(history.history['val_binary_accuracy'], 'r--')
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper right')
		plt.show()

		loss_array = history.history['loss']
		validation_loss_array = history.history['val_loss']
		epochs_array = np.arange(1, self.epochs+1)
		# plt.plot(loss_array, 'bo')
		plt.plot(epochs_array, loss_array, 'b', label='Train')
		plt.plot(epochs_array, validation_loss_array, 'r--', label='Test')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		# NOTE: this doesn't show the output layer
		plt.title('Loss-Epochs\nNetwork Architecture '+str(self.architecture))
		plt.legend(loc='upper right')
		plt.show()

	def k_fold(self, x, y, n_splits=5, show_single=True):
		start_time = time.time()
		kf = KFold(n_splits=n_splits, shuffle=True)
		scaler = StandardScaler()
		k_fold_history = []
		k_fold_mee = []
		for index, (train, test) in enumerate(kf.split(x)):
			model = self.new_model()
			x_train, y_train = x[train], y[train]
			x_test, y_test = x[test], y[test]
			x_train = scaler.fit_transform(x_train)
			x_test = scaler.transform(x_test)
			history, error = self.train_validate(
				x_train, y_train, x_test, y_test, model)
			k_fold_mee.append(error)
			k_fold_history.append(history)
			if show_single:
				print('#iteration n° %d\n' % (index+1))
				self.show_result(history)
        # add functionality that print mean over the k-fold
        # if show_mean :
		mean_mee = np.mean(k_fold_mee)
		print('time: %.2f' % (time.time()-start_time))
		print('Mean Euclidean error (over all evaluations): %.3f' % mean_mee)
		return k_fold_history, k_fold_mee
	
	def hp_tuning_GS(self, x, y, param_grid, folds=5, save=True, filename="NN_GS.csv"):
		start_time = time.time()
        # Parameters to be optimized can be choosen between the parameters of self.new_model and are
        # given through **kwargs as --> parameter=[list of values to try for tuning]
        # NOTE: batch_size and epochs can also be choosen
        # The CSV file with the result is saved inside the result/ folder
		estimator = KerasClassifier(self.new_model)
		print(param_grid)
		grid = GridSearchCV(
			estimator=estimator, param_grid=param_grid, return_train_score=True, cv=folds)
		print('\n\n\n\n')
		grid_fitted = grid.fit(x, y)
		means_test = grid_fitted.cv_results_['mean_test_score']
		std_test = grid_fitted.cv_results_['std_test_score']
		means_train = grid_fitted.cv_results_['mean_train_score']
		stds = grid_fitted.cv_results_['std_test_score']
		params = grid_fitted.cv_results_['params']
		for mean, stdev, param in zip(means_test, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))
		print('Best score obtained: %f \nwith param: %s' %
              (grid_fitted.best_score_, grid_fitted.best_params_))
		print('Total elapsed time: %.3f' % (time.time()-start_time))
		if save:
			dict_csv={}
			dict_csv.update({'Mean Score' : []})
			dict_csv.update({'Std Dev' : []})
			for key in params[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(params):
				for key in val:
					dict_csv[key].append((val[key]))
				dict_csv['Mean Score'].append(means_test[index])
				dict_csv['Std Dev'].append(std_test[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/result_encoded/"+filename),sep=',', index_label='Index')
		return grid_fitted

	def hp_tuning_RS(self, x, y, param_dist, iterations=10, folds=5, save=True, filename="NN_RS.csv"):
		start_time = time.time()
        # NOTE: continuous parameters should be given as a distribution for a proper random search
        # Distributions can be generated with scipy.stats module
        # For parameters that need to be explored in terms of order of magnitude loguniform distribution is recommended
		estimator = KerasClassifier(self.new_model)
		random_grid = RandomizedSearchCV(
		    estimator, param_dist, n_iter=iterations, cv=folds, return_train_score=True)
		grid_fitted = random_grid.fit(x, y)
		means_test = grid_fitted.cv_results_['mean_test_score']
		means_train = grid_fitted.cv_results_['mean_train_score']
		stds = grid_fitted.cv_results_['std_test_score']
		params = grid_fitted.cv_results_['params']
		for mean, stdev, param in zip(means_test, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))
		print('Best score obtained: %f \nwith param: %s' %
		      (grid_fitted.best_score_, grid_fitted.best_params_))
		print('Total elapsed time: %.3f' % (time.time()-start_time))
		if save:
			dict_csv={}
			dict_csv.update({'Score' : []})
			for key in params[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(params):
				for key in val:
					dict_csv[key].append((val[key]))
				dict_csv['Score'].append(means_test[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/result_encoded/"+filename),sep=',', index_label='Index')
		return grid_fitted
	
	def k_fold_param_func_minimize(self,x,y,n_splits=5):
		#NOTE:Data should be given already scaled
		kf = KFold(n_splits=n_splits, shuffle=True)
		def k_fold_param(params):
			#NOTE: params is a dictionary
			k_fold_loss = []
			k_fold_metric = []
			model = self.new_model(**params)
			starting_weights = model.get_weights()
			for index, (train, test) in enumerate(kf.split(x)):
				x_train, y_train = x[train], y[train]
				x_test, y_test = x[test], y[test]
				model.fit(x=x_train, y=y_train, epochs=self.epochs,
								shuffle=True, batch_size=self.batch_size, verbose=2)
				score = model.test_on_batch(x_test,y_test)
				k_fold_loss.append(score[0])
				k_fold_metric.append(score[1])
				model.set_weights(starting_weights)
			#NOTE: given that the score returned is used for Bayesian Optimization we return the mean
			# value of the loss function
			print("Mean value of the selected metric on test splits: %.3f" %np.mean(k_fold_metric))
			mean_loss = np.mean(k_fold_loss)
			print("Mean value of the selected metric on test splits: %.3f" %mean_loss)
			return {'loss': mean_loss, 'status': STATUS_OK}
		return k_fold_param
 
	def hp_tuning_BO(self, x, y, param_dist,iterations=10, save=True, filename='"NN_BO_noname.csv"'):
		start_time = time.time()
		# As in randomized search parameters should be passed as distributions.
		# For compatibility with HyperOpt is best to pass them using HyperOpt distributions
		objective_function = self.k_fold_param_func_minimize(x,y)
		trials = Trials()
		seed=int(time.time())
		best_param = fmin(objective_function, 
                      param_dist, 
                      algo=tpe.suggest, 
                      max_evals=iterations, 
                      trials=trials,
                      rstate=np.random.RandomState(seed)
					  )
		print('time: %.2f' % (time.time()-start_time))
		best_param_values = [val for val in best_param.values()]
		losses = [x['result']['loss'] for x in trials.trials]
		vals = [x['misc']['vals']for x in trials.trials]
		for val, loss in zip(vals,losses):
			print('Loss: %f   Param:%s' %(loss,val))
		best_param_values = [x for x in best_param.values()]
		print("Best loss obtained: %f\n with parameters: %s" % (min(losses), best_param_values))
		if save:
			dict_csv={}
			dict_csv.update({'Score' : []})
			for key in vals[0]:
				dict_csv.update({key : []})
			for index,val in enumerate(vals):
				for key in val:
					dict_csv[key].append((val[key])[0])
				dict_csv['Score'].append(losses[index])
			df = pd.DataFrame.from_dict(dict_csv, orient='columns')
			df.to_csv(path_or_buf=("./result/NN_class_ken/"+filename),sep=',', index_label='Index')
		return trials

def main():
	pass

if __name__ == '__main__':
	main()
