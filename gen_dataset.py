###
##generate dataset

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz
import datetime
import sklearn.preprocessing
from sklearn.utils import shuffle

def createDevAndTest(Dataset):
    shuffle(Dataset)
    Our_Train = Dataset[:-200]
    Our_Test = Dataset[-200:]
    pd.DataFrame(Our_Train).to_csv("Scrivania/dataset/OUR_TRAIN.csv",header = False)
    pd.DataFrame(Our_Test).to_csv("Scrivania/dataset/OUR_TEST.csv",header = False)
#     print (Dev)
#     print (Dev.shape)
    return True

def CSV_to_array(file_name, shuf=False, delimiter=',', comment='#'):
    dataframe = pd.read_csv(file_name, delimiter = delimiter, comment=comment)
    if shuf :
        dataframe = shuffle(dataframe)
        dataframe = dataframe.reset_index(drop=True)
    array = dataframe.values[:,1:]
    return array


def readFile(name):
    df = pd.read_csv(name, delimiter=',', comment='#', header=None)
#     if shuff:
#         df = shuffle(df)
#         df = df.reset_index(drop=True)
    df = df.astype(float)
    #in order to return numpy object
    df_numpy = df.values[:,1:]
    return df_numpy

def divide(Data, size = 2):
    X = Data[:, :-size]
    Y = Data[:, -size:]
    return X, Y

def main():
    df = readFile('Scrivania/dataset/ML-CUP19-TR.csv')
    createDevAndTest(df)


if __name__ == "__main__":
   main()
