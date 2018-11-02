import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import xgboost as xgb

trainX, trainy = dex.get_train()
validX, validy = dex.get_valid()
testX = dex.get_test()

def test_RF():
    res = []
    x, y = [], []
    print("Part1")
    for i in range(100, 400, 20):
        print("One test")
        nb_trees = np.random.randint(i, i + 20)
        model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(nb_trees)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning RF number of trees"])
    x, y = [], []
    print("Part2")
    for j in range(3, 93, 10):
        print("One test")
        nb_feat = np.random.randint(j, j + 10)
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=nb_feat)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(nb_feat)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning RF max_features"])
    x, y = [], []
    print("Part3")
    for i in range(100, 400, 20):
        print("One test")
        nb_trees = np.random.randint(i, i + 20)
        model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1)
        calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
        calibrated_model.fit(trainX, trainy)
        pred_valid = calibrated_model.predict_proba(validX)
        x.append(nb_trees)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning Calibrated RF number of trees"])
    x, y = [], []
    print("Part4")
    for j in range(6, 20, 2):
        print("One test")
        nb_feat = np.random.randint(j, j + 2)
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=nb_feat)
        calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
        calibrated_model.fit(trainX, trainy)
        pred_valid = calibrated_model.predict_proba(validX)
        x.append(nb_feat)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning Calibrated RF number of trees"])
    return res

def test_xgboost():
    res = []
    x, y = [], []
    print("Tuning XGB max_depth")
    for i in [1,2,3,4,6,8, 15, 30, 50, 100]:
        model = xgb.XGBClassifier(max_depth=i, n_estimators=100, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001, reg_alpha=0.003,
                                  min_child_weight=8, subsample=0.8, gamma=0.1)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i)
        y.append(score)
    res.append([x, y, "Tuning XGB max_depth"])
    x, y = [], []
    print("Tuning XGB number of trees")
    for i in [1,2,3,4,6,8]:
        model = xgb.XGBClassifier(max_depth=2, n_estimators=100 * i, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001, reg_alpha=0.003,
                                  min_child_weight=8, subsample=0.8, gamma=0.1)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i * 100)
        y.append(score)
    res.append([x, y, "Tuning XGB number of trees"])
    x, y = [], []
    print("Tuning XGB learning rate")
    for i in [1,3,5, 10, 50, 100, 500]:
        model = xgb.XGBClassifier(max_depth=2, n_estimators=100, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001 * i,
                                  reg_alpha=0.003,
                                  min_child_weight=8, subsample=0.8, gamma=0.1)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i * 0.001)
        y.append(score)
    res.append([x, y, "Tuning XGB learning rate"])
    x, y = [], []
    print("Tuning XGB alpha")
    for i in [1,2,3,4,6,8,10]:
        model = xgb.XGBClassifier(max_depth=2, n_estimators=100, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001,
                                  reg_alpha=0.001 * i,
                                  min_child_weight=8, subsample=0.8, gamma=0.1)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i * 0.001)
        y.append(score)
    res.append([x, y, "Tuning XGB alpha"])
    x, y = [], []
    print("Tuning XGB gamma")
    for i in [0,1,2,3,4,6,8]:
        model = xgb.XGBClassifier(max_depth=2, n_estimators=100, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001, reg_alpha=0.003,
                                  min_child_weight=8, subsample=0.8, gamma=0.1 * i)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i * 0.1)
        y.append(score)
    res.append([x, y, "Tuning XGB gamma"])
    x, y = [], []
    print("Tuning XGB min_child_weight")
    for i in [1,2,3,4,6,8,10]:
        model = xgb.XGBClassifier(max_depth=2, n_estimators=100, n_jobs=100, random_state=5,
                                  objective='multi:softprob', learning_rate=0.001, reg_alpha=0.003,
                                  min_child_weight=i, subsample=0.8, gamma=0.1 * i)
        model.fit(trainX, trainy, eval_metric=evaluation)
        pred_valid = model.predict_proba(validX)
        score = evaluation(validy, pred_valid)
        print(score)
        x.append(i)
        y.append(score)
    res.append([x, y, "Tuning XGB min_child_weight"])
    return res

def plot_tuning(data):
    for i, x in enumerate(data):
        f = plt.figure(i)
        plt.plot(x[0], x[1], "r", label='train (all samples)')
        plt.title(x[2])
        plt.xlabel("Value of parameter")
        plt.ylabel("Evaluation on valid set")
        plt.legend()
        f.show()

    input()

# plot_tuning(test_RF())
plot_tuning(test_xgboost())
