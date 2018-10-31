import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *

trainX, trainy = dex.get_train()
validX, validy = dex.get_valid()
testX = dex.get_test()

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def classif_randomForest(trainX, trainy, validX, validy, testX, name='../randomForest_submission.csv'):
    model = RandomForestClassifier(n_estimators=120, n_jobs=-1, oob_score=True)
    calibrated_model = CalibratedClassifierCV(model, 'isotonic')
    calibrated_model.fit(trainX, trainy)
    print("Model trained")


    #     pred_valid_normal = model.predict(validX)
    pred_valid = calibrated_model.predict_proba(validX)
    print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
#     print(mtc.accuracy_score(validy, pred_valid_normal))

    testProbas = calibrated_model.predict_proba(testX)
    saveResult(testProbas, name)

from sklearn.svm import SVC

def classif_SVM(trainX, trainy, validX, validy, testX, name='../svm_submission.csv'):
    model = SVC(probability=True)
    model.fit(trainX, trainy)
    print("Model trained")

#     pred_valid_normal = model.predict(validX)
    pred_valid = model.predict_proba(validX)
    print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
#     print(mtc.accuracy_score(validy, pred_valid_normal))

    testProbas = model.predict_proba(testX)
    saveResult(testProbas, name)


def test_RF(trainX, trainy, validX, validy):
    eval_valid_nbTree = [[],[]]
    eval_valid_nbFeature = [[],[]]
    for i in range(100, 400, 20):
        nb_trees = np.random.randint(i, i + 20)
        model = RandomForestClassifier(n_estimators=nb_trees, n_jobs=-1)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        eval_valid_nbTree[0].append(nb_trees)
        eval_valid_nbTree[1].append(evaluation(validy, pred_valid))
    for j in range(5, 20, 2):
        nb_feat = np.random.randint(j, j + 2)
        model = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_features=nb_feat)
        model.fit(trainX, trainy)
        pred_valid = model.predict_proba(validX)
        eval_valid_nbFeature[0].append(nb_feat)
        eval_valid_nbFeature[1].append(evaluation(validy, pred_valid))
    return eval_valid_nbTree, eval_valid_nbFeature

import matplotlib.pyplot as plt

def plot_testRF(score_nbTree, score_nbFeat):
    f1 = plt.figure(1)
    plt.plot(score_tree[0], score_tree[1], "r", label='train (all samples)')
    plt.title("Score different parameters RF")
    plt.xlabel("Value of parameter")
    plt.ylabel("Evaluation on valid set")
    plt.legend()
    plt.ylim((0, 1))
    f1.show()

    f2 = plt.figure(2)
    plt.plot(score_maxFeat[0], score_maxFeat[1], "g", label='valid (all samples)')
    plt.title("Score different parameters RF")
    plt.xlabel("Value of parameter")
    plt.ylabel("Evaluation on valid set")
    plt.legend()
    plt.ylim((0, 1))
    f2.show()

    input()

classif_randomForest(trainX, trainy, validX, validy, testX)
# classif_SVM(trainX, trainy, validX, validy, testX)
# score_tree, score_maxFeat = test_RF(trainX, trainy, validX, validy)
# plot_testRF(score_tree, score_maxFeat)
