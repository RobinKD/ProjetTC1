import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import xgboost as xgb


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

    valid_RF = calibrated_model.predict_proba(validX)
    valid_XGB = model_XGB.predict_proba(validX)

    res_RF = model_RF.predict_proba(testX)
    res_XGB = model_XGB.predict_proba(testX)
    for x in [y / 10.0 for y in range(1, 10)]:
        combres = (x * res_RF + (1 - x) * res_XGB) / 2
        if not len(validy) == 0:
            pred_valid = (x * valid_RF + (1 - x) * valid_XGB) / 2
            print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(combres, "../combined_{.1}RF_{.1}XGB.csv".format(x, 1 - x))



RF_model = RandomForestClassifier(n_estimators=350, max_features=15, n_jobs=-1, oob_score=True,
                                  random_state=5)
# XGB Jiaxin's parameters
# XGB_model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=700, gamma=0, max_depth=7,
#                               min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
#                               n_jobs=100, random_state=27, objective='multi:softprob')
XGB_model = xgb.XGBClassifier(max_depth=15, n_estimators=800, n_jobs=-1, random_state=5,
                              objective='multi:softprob', learning_rate=0.1, reg_alpha=0.003,
                              min_child_weight=3, subsample=0.8, gamma=0)

# res1 = classif_randomForest(RF_model)
# res2 = classif_xgboost(XGB_model)

averaged_XGB_RF(RF_model, XGB_model)

