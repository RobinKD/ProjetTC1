import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

trainX, trainy = dex.get_train()
validX, validy = dex.get_valid()
testX = dex.get_test()

def classif_randomForest(model, name='../randomForest_submission.csv'):
    """
    Train a calibrated RF, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
    calibrated_model.fit(trainX, trainy)
    print("Model trained")

    if not len(validy) == 0:
        pred_valid = calibrated_model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))

    testProbas = calibrated_model.predict_proba(testX)
    saveResult(testProbas, name)


def classif_SVM(name='../svm_submission.csv'):
    """
    Train a SVM, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    model = SVC(probability=True)
    model.fit(trainX, trainy)
    print("Model trained")

    if not len(validy) == 0:
        pred_valid = model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))

    testProbas = model.predict_proba(testX)
    saveResult(testProbas, name)

def classif_xgboost(model, name='../xgboost_submission.csv'):
    """
    Train an XGBClassifier, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    print("Creation model")
    model.fit(trainX, trainy, eval_metric=evaluation)
    print("Model trained")


    if not len(validy) == 0:
        pred_valid = model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))

    testProbas = model.predict_proba(testX)
    saveResult(testProbas, name)

def classif_MLP(model, name='../MLP_submission.csv'):
    """
    Train a calibrated RF, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    model.fit(trainX, trainy)
    print("Model trained")

    if not len(validy) == 0:
        pred_valid = model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))

    testProbas = model.predict_proba(testX)
    saveResult(testProbas, name)


def averaged_XGB_RF(model_RF, model_XGB):
    """
    A try to average RF model and XGB model with different weights
    to see if it can be better
    """
    calibrated_model = CalibratedClassifierCV(model_RF, 'isotonic', 5)
    calibrated_model.fit(trainX, trainy)
    print("Calibrated RF trained")

    model_XGB.fit(trainX, trainy, eval_metric=evaluation)
    print("XGB model trained")

    if not len(validy) == 0:
        valid_RF = calibrated_model.predict_proba(validX)
        valid_XGB = model_XGB.predict_proba(validX)

    res_RF = calibrated_model.predict_proba(testX)
    saveResult(res_RF, "../randomForest_submission.csv")
    res_XGB = model_XGB.predict_proba(testX)
    saveResult(res_XGB, "../XGB_submission.csv")
    for x in [y / 100.0 for y in range(30, 70)]:
        combres = (x * res_RF + (1 - x) * res_XGB)
        if not len(validy) == 0:
            pred_valid = (x * valid_RF + (1 - x) * valid_XGB)
            print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(combres, "../combined_csv/combined_{:1.2f}RF_{:1.2f}XGB.csv".format(x, 1 - x))

def averaged_3models(model_RF, model_XGB, model_MLP):
    """
    A try to average RF model and XGB model with different weights
    to see if it can be better
    """
    calibrated_model = CalibratedClassifierCV(model_RF, 'isotonic', 5)
    calibrated_model.fit(trainX, trainy)
    print("Calibrated RF trained")

    model_XGB.fit(trainX, trainy, eval_metric=evaluation)
    print("XGB model trained")

    model_MLP.fit(trainX, trainy)
    print("MLP model trained")

    if not len(validy) == 0:
        valid_RF = calibrated_model.predict_proba(validX)
        valid_XGB = model_XGB.predict_proba(validX)
        valid_MLP = model_MLP.predict_proba(validX)

    res_RF = calibrated_model.predict_proba(testX)
    saveResult(res_RF, "../randomForest_submission.csv")
    res_XGB = model_XGB.predict_proba(testX)
    saveResult(res_XGB, "../XGB_submission.csv")
    res_MLP = model_MLP.predict_proba(testX)
    saveResult(res_MLP, "../MLP_submission.csv")
    # for x in [y / 100.0 for y in range(80, 99, 2)]:
    #     combres = x * (0.5 * res_RF + 0.5 * res_XGB) + (1 - x) * res_MLP
    #     if not len(validy) == 0:
    #         pred_valid = x * (0.5 * valid_RF + 0.5 * valid_XGB) + (1 - x) * valid_MLP
    #         print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
    #     saveResult(combres, "../combined_csv/combined_{:1.1f}RF_XGB_{:1.1f}MLP.csv".format(x, 1-x))
    combres = 0.88 * (0.5 * res_RF + 0.5 * res_XGB) + (0.12) * res_MLP
    saveResult(combres, "../combined_csv/combined_{:1.2f}RF_XGB_{:1.2f}MLP.csv".format(0.88, 0.12))

RF_model = RandomForestClassifier(n_estimators=350, max_features=15, n_jobs=3, oob_score=True,
                                  random_state=5)
# XGB Jiaxin's parameters
# XGB_model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=700, gamma=0, max_depth=7,
#                               min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
#                               n_jobs=100, random_state=27, objective='multi:softprob')
XGB_model = xgb.XGBClassifier(max_depth=15, n_estimators=800, n_jobs=3, random_state=5,
                              objective='multi:softprob', learning_rate=0.1, reg_alpha=0.003,
                              min_child_weight=3, subsample=0.8, gamma=0)

MLP_model = MLPClassifier((75,), alpha=0.00001, learning_rate_init=0.0005, max_iter=14,
                          tol=0.005, random_state=5)

# classif_randomForest(RF_model)
# classif_xgboost(XGB_model)
# classif_MLP(MLP_model)

# averaged_XGB_RF(RF_model, XGB_model)
averaged_3models(RF_model, XGB_model, MLP_model)
