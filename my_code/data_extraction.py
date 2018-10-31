import numpy as np
import pandas as pd

panda_dataset = pd.read_csv('../train.csv')
panda_testset = pd.read_csv('../test.csv')
dataset = panda_dataset.values[:,1:]
testX = panda_testset.values[:, 1:]

import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


X, y = dataset[:, :-1], dataset[:, -1:]

trainX, validX, trainy, validy = train_test_split(X, y, test_size=0.0, random_state=5)
trainy, validy = trainy.ravel(), validy.ravel()

def preproc(data):
    d = np.array(data, dtype='float64')
    d = prep.normalize(d, norm='l2', axis=0)
    d = prep.scale(d, axis=0)
    return d


def get_train():
    train_X = preproc(trainX)
    return train_X, trainy

def get_valid():
    if len(validy) == 0:
        return [], []
    else:
        valid_X = preproc(validX)
        return preproc(valid_X), validy

def get_test():
    test_X = preproc(testX)
    return preproc(test_X)
