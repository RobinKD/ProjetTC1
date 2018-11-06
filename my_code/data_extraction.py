import numpy as np
import pandas as pd

panda_dataset = pd.read_csv('../train.csv')
panda_testset = pd.read_csv('../test.csv')
dataset = panda_dataset.values[:,1:]
testX = panda_testset.values[:, 1:]

import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split


X, y = dataset[:, :-1], dataset[:, -1:]

trainX, validX, trainy, validy = train_test_split(X, y, test_size=0.1, random_state=5)
trainy, validy = trainy.ravel(), validy.ravel()

def preproc(data):
    """
    To normalize and scale dataset given in parameters
    """
    d = np.array(data, dtype='float64')
    d = prep.normalize(d, norm='l2', axis=0)
    d = prep.scale(d, axis=0)
    return d

def encode_label(y):
    """
    To change string labels to ints for simplicity
    """
    return prep.LabelEncoder().fit_transform(y)

def get_train():
    """
    return preprocessed trainset and encoded train labels
    """
    train_X = preproc(trainX)
    train_y = encode_label(trainy)
    return train_X, train_y

def get_valid():
    """
    return preprocessed validset and encoded valid labels of size of valid set
    bigger than 0, otherwise empty lists
    """
    if len(validy) == 0:
        return [], []
    else:
        valid_X = preproc(validX)
        valid_y = encode_label(validy)
        return valid_X, valid_y

def get_test():
    """
    return preprocessed testset
    """
    test_X = preproc(testX)
    return test_X
