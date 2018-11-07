import numpy as np
import pandas as pd

import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def get_data(traincsv, testcsv):
    """
    Get data from relative paths of csv files for train and test
    """
    panda_dataset = pd.read_csv(traincsv)
    panda_testset = pd.read_csv(testcsv)
    dataset = panda_dataset.values[:,1:]
    X, y = dataset[:, :-1], dataset[:, -1:]
    testX = panda_testset.values[:, 1:]
    return X, y, testX

X, y, testX = get_data("../train.csv", "../test.csv")

def data_validation(X, y):
    """
    Return train and validation data then train and validation labels
    to train models and test logloss locally
    Stratified with one bin to keep approximately the same class
    distribution in train and test
    """
    Sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=5)
    for train_index, test_index in Sss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainX, validX = X[train_index], X[test_index]
        trainy, validy = y[train_index], y[test_index]
    return trainX, validX, trainy.ravel(), validy.ravel()

def data_submit(X, y):
    """
    All data in train for better scores while submitting
    """
    return X, [], y.ravel(), []

def preproc(X):
    """
    To scale dataset given in parameters
    PCA abandoned because it worsened results => all features are important
    Normalization abandoned because unstable
    """
    d = np.array(X, dtype='float64')
    d = prep.scale(d, axis=0)
    return d

def encode_label(y):
    """
    To change string labels to ints for simplicity
    """
    return prep.LabelEncoder().fit_transform(y)

def get_preproc_train(trainX, trainy):
    """
    return preprocessed trainset and encoded train labels
    """
    train_X = preproc(trainX)
    train_y = encode_label(trainy)
    return train_X, train_y

def get_preproc_valid(validX, validy):
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

def get_preproc_test(testX):
    """
    return preprocessed testset
    """
    test_X = preproc(testX)
    return test_X
