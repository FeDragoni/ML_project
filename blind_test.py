import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
import csv
import gen_dataset
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import csv
import time
import Functions


train_dataset = gen_dataset.CSV_to_array('./dataset/ML-CUP19-TR.csv', shuf=True, delimiter=',',header=None)
test_dataset = gen_dataset.CSV_to_array('./dataset/ML-CUP19-TS.csv', shuf=False, delimiter=',',header=None)
x_train,y_train=gen_dataset.divide(train_dataset)
x_test = test_dataset.copy()
print(np.round(x_test,3))
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x train: ",x_train.shape)
print("y train: ",y_train.shape)
print("x test: ",x_test.shape)

myRF = RandomForestRegressor(n_estimators=700,max_features=3,bootstrap=False)
myRF.fit(x_train, y_train)
y_pred = myRF.predict(x_test)

blind_test_df = pd.DataFrame(y_pred)
blind_test_df.index = np.arange(1, len(blind_test_df) + 1)
blind_test_df.to_csv("./dataset/TrotKen.csv",header = False)
print(myRF.predict(x_test[[0,1,-1]]))