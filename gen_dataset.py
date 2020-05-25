import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz
import datetime
import sklearn.preprocessing
from sklearn.utils import shuffle

def gen_Train_Test(Dataset):
    shuffle(Dataset)
    Our_Train = Dataset[:-199]
    Our_Test = Dataset[-201:]
    pd.DataFrame(Our_Test).to_csv("./dataset/OUR_TEST.csv",header = False)
    return True

def CSV_to_array(file_name, shuf=True, delimiter=',', comment='#',header=None):
    dataframe = pd.read_csv(file_name, delimiter = delimiter, comment=comment, header=header)
    dataframe = dataframe.astype(float)
    if shuf :
        dataframe = shuffle(dataframe)
        dataframe = dataframe.reset_index(drop=True)
    array = dataframe.values[:,1:]
    return array


def divide(Data, size = 2):
    X = Data[:, :-size]
    Y = Data[:, -size:]
    return X, Y

def main():
    df = CSV_to_array('./dataset/ML-CUP19-TR.csv')
    print(df.shape)
    #gen_Train_Test(df)
    train_dataset = CSV_to_array('./dataset/OUR_TRAIN.csv')
    print(train_dataset.shape)
    x_train,y_train = divide(train_dataset)
    print("x shape: ", x_train.shape)
    print("y shape: ", y_train.shape)
    test_dataset = CSV_to_array('./dataset/OUR_TEST.csv')
    x_test,y_test = divide(test_dataset)
    print("x shape: ", x_test.shape)
    print("y shape: ", y_test.shape)
