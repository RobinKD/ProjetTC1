import numpy as np
import pandas as pd

panda_dataset = pd.read_csv('../train.csv')
panda_testset = pd.read_csv('../test.csv')
dataset = panda_dataset.values
dataset = dataset[:,1:]
test = panda_testset.values
testX = test[:, 1:]

import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split

X, y = dataset[:, :-1], dataset[:, -1:]

trainX, validX, trainy, validy = train_test_split(X, y, test_size=0.1, random_state=5)

# np.random.shuffle(dataset)
# endTrain = round(0.9 * len(dataset))
# train = dataset[:endTrain, :]
# valid = dataset[endTrain:, :]

# trainy = train[:, -1:].reshape(-1)
# trainX = train[:, :-1]

# validy = valid[:, -1:].reshape(-1)
# validX = valid[:, :-1]

def preproc(data):
    d = prep.normalize(data, norm='l1', axis=0)
    d = prep.scale(d, axis=0)
    return d

trainX = preproc(trainX)
validX = preproc(validX)
testX = preproc(testX)

def get_train():
    return trainX, trainy

def get_valid():
    return validX, validy

def get_test():
    return testX
