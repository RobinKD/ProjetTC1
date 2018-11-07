import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore") # To ignore convergence warnings from MLPClassifier

def test_RF(trainX, trainy, validX, validy):
    """
    Unit tests to search good values for parameters
    """
    print("Test parameters RF")
    res = []
    x, y = [], []
    print("Part1 - Non calibrated")
    for i in range(100, 400, 20):
        model = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(nb_trees)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning RF number of trees"])
    x, y = [], []
    for j in range(3, 93, 10):
        nb_feat = np.random.randint(j, j + 10)
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=j)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(nb_feat)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning RF max_features"])
    x, y = [], []
    print("Part2 - Calibrated")
    for i in range(100, 400, 20):
        nb_trees = np.random.randint(i, i + 20)
        model = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
        calibrated_model.fit(trainX, trainy)
        pred_valid = calibrated_model.predict_proba(validX)
        x.append(nb_trees)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning Calibrated RF number of trees"])
    x, y = [], []
    for j in range(6, 20, 2):
        nb_feat = np.random.randint(j, j + 2)
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=j)
        calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
        calibrated_model.fit(trainX, trainy)
        pred_valid = calibrated_model.predict_proba(validX)
        x.append(nb_feat)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning Calibrated RF number of trees"])
    return res

def test_xgboost(trainX, trainy, validX, validy):
    """
    Tuning not calibrated XGBClassifier because not much different, faster, and improvement
    stable when calibrated, it does not really depend on parameters.
    Some parameters already have "good" values found online, or default values.
    """
    print("Test parameters XGBClassifier")
    res = []
    x, y = [], []
    print("Tuning XGB max_depth")
    for i in [1,2,3,4,6,8, 15, 30, 50, 100]:
        model = xgb.XGBClassifier(max_depth=i, n_estimators=100, objective='multi:softprob',
                                  learning_rate=0.001, reg_alpha=0.003,
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

def test_MLP(trainX, trainy, validX, validy):
    """
    Unit tests for MLPClassifier
    Bagged because it improves results
    """
    print("Test parameters MLPClassifier")
    res = []
    x, y = [], []
    print("Tuning max_iter")
    for i in range(2, 20, 2):
        model = MLPClassifier(max_iter=i)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning max_iter"])
    x, y = [], []
    print("Tuning hidden layers")
    for i in [(100,60,20), (50,30,10), (100,50), (50,25), (100,), (75,),(50,), (30,), (20,)]:
        model = MLPClassifier(hidden_layer_sizes=i, max_iter=50)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i[0])
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning hidden layers"])
    x, y = [], []
    print("Tuning regularization")
    for i in [j / 10 ** 6 for j in [1, 5, 10, 50, 100]]:
        model = MLPClassifier(max_iter=50, alpha=i)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning regularization"])
    x, y = [], []
    print("Tuning learning rate init")
    for i in [j / 10**4 for j in [1, 5, 10, 50, 100]]:
        model = MLPClassifier(max_iter=50, learning_rate_init=i)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning learning rate init"])
    x, y = [], []
    print("Tuning tol")
    for i in [j / 10 ** 4 for j in [1, 5, 10, 50, 100]]:
        model = MLPClassifier(max_iter=50, tol=i)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning tol"])
    x, y = [], []
    print("Tuning activation function")
    for i in ['identity', 'logistic', 'tanh', 'relu']:
        model = MLPClassifier(max_iter=20, activation=i)
        bagmodel = BaggingClassifier(model)
        bagmodel.fit(trainX, trainy)
        pred_valid = bagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning activation function"])
    x, y = [], []
    print("Tuning solver")
    for i in ['lbfgs', 'sgd', 'adam']:
        model = MLPClassifier(max_iter=20, hidden_layer_sizes=(75), solver=i)
        bagmodel = BaggingClassifier(model)
        bagbagmodel.fit(trainX, trainy)
        pred_valid = bagbagmodel.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning solver"])
    return res

def test_Adaboost(trainX, trainy, validX, validy):
    """
    Unit tests for Adaboost
    Not calibrated because it worsen results
    """
    print("Test parameters AdaBoostClassifier")
    res = []
    rfmodel = RandomForestClassifier(n_estimators=350, max_features=15)
    x, y = [], []
    print("Test not calibrated")
    for i in range(10, 410, 50):
        model = AdaBoostClassifier(base_estimator = rfmodel, n_estimators=i)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Test not calibrated"])
    x, y = [], []
    print("Tuning n_estimators")
    for i in range(10, 100, 10):
        model = AdaBoostClassifier(base_estimator = rfmodel, n_estimators=i)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning n_estimators"])
    x, y = [], []
    print("Tuning learning rate")
    for i in [j / 10 for j in range(1, 11)]:
        model = AdaBoostClassifier(base_estimator = rfmodel, learning_rate=i)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning learning rate"])
    x, y = [], []
    print("Tuning n_estimators & learning rate")
    for i in range(50, 90, 5):
        for j in [k / 20 for k in range(2, 6)]:
            model = AdaBoostClassifier(base_estimator = rfmodel, n_estimators=i, learning_rate=j)
            model.fit(trainX, trainy)
            pred_valid = model.predict_proba(validX)
            x.append(str(i) + str(j))
            y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning n_estimators & learning rate"])
    return res

def test_extratree(trainX, trainy, validX, validy):
    """
    Unit tests for extra tree.
    """
    print("Test parameters ExtraTreesClassifier")
    res = []
    x, y = [], []
    print("Test not calibrated")
    for i in range(210, 510, 50):
        model = ExtraTreesClassifier(n_estimators=i)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Test not calibrated"])
    x, y = [], []
    print("Test calibrated")
    for i in range(210, 510, 50):
        model = ExtraTreesClassifier(n_estimators=i)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Test calibrated"])
    x, y = [], []
    print("Tuning n_estimators")
    for i in range(210, 510, 50):
        model = ExtraTreesClassifier(n_estimators=i)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning n_estimators"])
    x, y = [], []
    print("Tuning max_depth")
    for i in [1, 3, 5, 10, 15, 25, None]:
        model = ExtraTreesClassifier(max_depth=i)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning max_depth"])
    x, y = [], []
    print("Tuning max_features")
    for i in [1, 3, 5, 8, 15, 30, 50, 93]:
        model = ExtraTreesClassifier(max_features=i)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning max_features"])
    x, y = [], []
    print("Tuning boostrap")
    for i in [True, False]:
        model = ExtraTreesClassifier(bootstrap=i)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning boostrap"])
    x, y = [], []
    print("Tuning oob")
    for i in [True, False]:
        model = ExtraTreesClassifier(oob_score=i, bootstrap=True)
        calib = CalibratedClassifierCV(model, 'isotonic', 3)
        calib.fit(trainX, trainy)
        pred_valid = calib.predict_proba(validX)
        x.append(i)
        y.append(evaluation(validy, pred_valid))
    res.append([x, y, "Tuning oob"])
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


X, y, test_X = dex.get_data("../train.csv", "../test.csv")
train_X, valid_X, train_y, valid_y = dex.data_validation(X, y)
trainX, trainy = dex.get_preproc_train(train_X, train_y)
validX, validy = dex.get_preproc_valid(valid_X, valid_y)
testX = dex.get_preproc_test(test_X)

plot_tuning(test_RF(trainX, trainy, validX, validy))
plot_tuning(test_xgboost(trainX, trainy, validX, validy))
plot_tuning(test_MLP(trainX, trainy, validX, validy))
plot_tuning(test_Adaboost(trainX, trainy, validX, validy))
plot_tuning(test_extratree(trainX, trainy, validX, validy))
