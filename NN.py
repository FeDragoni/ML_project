from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.constraints import max_norm
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import pandas as pd

class NeuralNetwork():
    def __init__(self, architecture, input_dimension, output_dimension, epochs, batch_size, dropout_rate_input=0, dropout_rate_hidden=0):
        self.architecture = architecture    #number of units in each layer                    
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate_input = dropout_rate_input
        self.dropout_rate_hidden = dropout_rate_hidden
        self.last_model_compiled = Sequential()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
    
    def new_model(self, activation='relu', kernel_initializer='normal',
                kernel_constraint=4.0, loss_function=keras.losses.mean_squared_error,
                lr=0.01, mom=0.0, nesterov=False):
        model=Sequential()
        # add input layer
        if self.dropout_rate_input>0:
            model.add(Dropout(self.dropout_rate_input))
        model.add(Dense(kernel_initializer= kernel_initializer, kernel_constraint=max_norm(kernel_constraint),
                    units=self.architecture[0], activation=activation, input_dim=self.input_dimension))
        # add hidden nodes, w/ dropout if given
        for node in self.architecture[1:]:
            if self.dropout_rate_hidden>0:
                model.add(Dropout(self.dropout_rate_hidden))
            model.add(Dense(kernel_initializer= kernel_initializer, kernel_constraint=max_norm(kernel_constraint),
                    units=self.architecture[0], activation=activation))
        # output layer and model compilation NOTE: activation function of last layer need some considerations (e.g. using
        # a linear activation function for binary classification may not be the best choice)
        model.add(Dense(units=self.output_dimension,activation='sigmoid'))
        #NOTE: maybe it would be better to choose 'binary_accuracy' to evaluate model performance
        if classification:
            metric='accuracy'
            else
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nesterov), 
                        loss=loss_function, metrics=['accuracy'])
        self.last_model_compiled=model
        return model

    def train_validate(self, x_train, y_train, x_val, y_val, last_model=True, model=None):
        start_time = time.time()
        if last_model:
            model=self.last_model_compiled
        else:
            model=model
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        history = model.fit(x=x_train, y=y_train, epochs=self.epochs,
                            shuffle=True, batch_size=self.batch_size,
                            validation_data=(x_val,y_val), verbose=2)
        y_pred = model.predict(x_val)
        mean_euclidean_error = 0
        for pred_val, actual_val in zip (y_pred, y_val):
            mean_euclidean_error += norm(pred_val - actual_val)
        mean_euclidean_error=mean_euclidean_error/len(y_val)
        print('\ntime: %.3f s' %(time.time()-start_time))
        print('Mean Euclidean error: %.3f' %mean_euclidean_error)
        print('y_pred (%f) and y_val (%f) length' %(len(y_pred), len(y_val)))
        return history, mean_euclidean_error

    def show_result(self, history):   #should we also print accuracy?
        loss_array = history.history['loss']
        validation_loss_array = history.history['val_loss']
        epochs_array = np.arange(1, self.epochs+1)
        #plt.plot(loss_array, 'bo')
        plt.plot(epochs_array, loss_array, label='Training error')
        plt.plot(epochs_array, validation_loss_array, label='Validation error')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss-Epochs\nNetwork Architecture '+str(self.architecture))  #NOTE: this doesn't show the output layer
        plt.legend(loc='upper right')
        plt.show()
    
    def k_fold(self,x, y, n_splits=5, show_single=True):
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
            history, error = self.train_validate(x_train, y_train, x_test, y_test, model)
            k_fold_mee.append(error)
            k_fold_history.append(history)
            if show_single :
                print('#iteration n° %d\n' %(index+1))
                self.show_result(history)
        #add functionality that print mean over the k-fold
        #### if show_mean :
        mean_mee = np.mean(k_fold_mee)
        print('time: %.2f' %(time.time()-start_time))
        print('Mean Euclidean error (over all evaluations): %.3f' %mean_mee)
        return k_fold_history, k_fold_mee

    def hp_tuning_GS(self, x, y, folds=5, save=True, filename="NN_GS.csv", **kwargs):
        start_time=time.time()
        # Parameters to be optimized can be choosen between the parameters of self.new_model and are 
        # given through **kwargs as --> parameter=[list of values to try for tuning]
        # NOTE: batch_size and epochs can also be choosen
        #The CSV file with the result is saved inside the result/ folder
        estimator = KerasClassifier(self.new_model)
        param_grid = kwargs
        print(param_grid)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, return_train_score=True, cv=folds)
        print('\n\n\n\n')
        grid_fitted = grid.fit(x, y)
        means_test = grid_fitted.cv_results_['mean_test_score']
        means_train = grid_fitted.cv_results_['mean_train_score']
        stds = grid_fitted.cv_results_['std_test_score']
        params = grid_fitted.cv_results_['params']
        for mean, stdev, param in zip(means_test, stds, params):
	        print("%f (%f) with: %r" % (mean, stdev, param))
        print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
        print('Total elapsed time: %.3f' %(time.time()-start_time))
        if save:
            df=pd.DataFrame(zip(means_test,means_train,means_train,params))
            df = df.rename(index=str, columns={0: "mean Validation Score",1:"mean Train Score",2: "Parameters"})
            df.to_csv("./result/"+filename)
        return grid_fitted 

    def hp_tuning_RS(self, x, y, iterations=10, folds=5,save=True, filename="NN_RS.csv",**kwargs):
        start_time=time.time()
        #NOTE: continuous parameters should be given as a distribution for a proper random search
        #Distributions can be generated with scipy.stats module
        #For parameters that need to be explored in terms of order of magnitude loguniform distribution is recommended
        estimator = KerasClassifier(self.new_model)
        param_distr = kwargs
        random_grid = RandomizedSearchCV(estimator,param_disrt,n_iter=iterations,cv=folds, return_train_score=True)
        grid_fitted = random_grid.fit(x,y)
        means_test = grid_fitted.cv_results_['mean_test_score']
        means_train = grid_fitted.cv_results_['mean_train_score']
        stds = grid_fitted.cv_results_['std_test_score']
        params = grid_fitted.cv_results_['params']
        for mean, stdev, param in zip(means_test, stds, params):
	        print("%f (%f) with: %r" % (mean, stdev, param))
        print('Best score obtained: %f \nwith param: %s' %(grid_fitted.best_score_, grid_fitted.best_params_))
        print('Total elapsed time: %.3f' %(time.time()-start_time))
        if save:
            df=pd.DataFrame(zip(means_test,means_train,means_train,params))
            df = df.rename(index=str, columns={0: "mean Validation Score",1:"mean Train Score",2: "Parameters"})
            df.to_csv("./result/"+filename)
        return grid_fitted

    def hp_tuning_BO(self,x,y,):
        return



def main():
    pass

if __name__ == '__main__':
    main()