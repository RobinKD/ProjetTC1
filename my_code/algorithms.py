import numpy as np
import data_extraction as dex
import data_engeneering as den
from res_treatment import *
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

def classif_randomForest(trainX, validX, trainy, validy, testX, model, name='../randomForest_submission.csv'):
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
        saveResult(pred_valid, name[:-4] + "_valid.csv")

    testProbas = calibrated_model.predict_proba(testX)
    saveResult(testProbas, name)


def classif_SVM(trainX, validX, trainy, validy, testX, name='../svm_submission.csv'):
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
        saveResult(pred_valid, name[:-4] + "_valid.csv")

    testProbas = model.predict_proba(testX)
    saveResult(testProbas, name)

def classif_xgboost(trainX, validX, trainy, validy, testX, model, name='../xgboost_submission.csv'):
    """
    Train an XGBClassifier, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    print("Creation model")
    calibrated_model = CalibratedClassifierCV(model, 'isotonic', 2)
    calibrated_model.fit(trainX, trainy)
    print("Model trained")


    if not len(validy) == 0:
        pred_valid = calibrated_model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(pred_valid, name[:-4] + "_valid.csv")

    testProbas = calibrated_model.predict_proba(testX)
    saveResult(testProbas, name)

def classif_MLP(trainX, validX, trainy, validy, testX, model, name='../MLP_submission.csv'):
    """
    Train a MLPClassifier, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    bagmodel = BaggingClassifier(model)
    bagmodel.fit(trainX, trainy)
    print("Model trained")

    if not len(validy) == 0:
        pred_valid = bagmodel.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(pred_valid, name[:-4] + "_valid.csv")

    testProbas = bagmodel.predict_proba(testX)
    saveResult(testProbas, name)

def classif_ET(trainX, validX, trainy, validy, testX, model, name='../ET_submission.csv'):
    """
    Train an ExtraTreesClassifier, test it on valid set if it exists,
    and save probabilities of testset in a csv file for
    submission
    """
    calibrated_model = CalibratedClassifierCV(model, 'isotonic', 5)
    calibrated_model.fit(trainX, trainy)
    print("Model trained")

    if not len(validy) == 0:
        pred_valid = calibrated_model.predict_proba(validX)
        print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(pred_valid, name[:-4] + "_valid.csv")

    testProbas = calibrated_model.predict_proba(testX)
    saveResult(testProbas, name)


def averaged_2models(trainX, validX, trainy, validy, testX, model1, model2, sub1, sub2):
    """
    A try to average two models with different weights
    to see if it can be better, by grid search
    sub1 and sub2 are parameters to change name of submission file
    Better to use averaging_probas to avoid retraining classifiers
    """
    if isinstance(model1, MLPClassifier) or isinstance(model1, xgb.XGBClassifier):
        calib1 = BaggingClassifier(model1)
    else:
        calib1 = CalibratedClassifierCV(model1, 'isotonic', 3)

    calib1.fit(trainX, trainy)
    print("model1 trained")

    if isinstance(model2, MLPClassifier) or isinstance(model2, xgb.XGBClassifier):
        calib2 = BaggingClassifier(model2)
    else:
        calib2 = CalibratedClassifierCV(model2, 'isotonic', 3)
    calib2.fit(trainX, trainy)
    print("model2 trained")

    if not len(validy) == 0:
        valid1 = calib1.predict_proba(validX)
        print("Evaluation model1(kaggle) of validation set :", evaluation(validy, valid1))
        valid2 = calib2.predict_proba(validX)
        print("Evaluation model2(kaggle) of validation set :", evaluation(validy, valid2))

    res1 = calib1.predict_proba(testX)
    saveResult(res1, "../"+sub1+"_submission.csv")
    res2 = calib2.predict_proba(testX)
    saveResult(res2, "../"+sub2+"_submission.csv")
    for x in [y / 10.0 for y in range(1,10)]:
        combres = (x * res1 + (1 - x) * res2)
        if not len(validy) == 0:
            pred_valid = (x * valid1 + (1 - x) * valid2)
            print("Evaluation (kaggle) of validation set :", evaluation(validy, pred_valid))
        saveResult(combres, "../combined_csv/combined_{:1.2f}{}_{:1.2f}{}.csv".format(x, sub1, 1 - x, sub2))

def run_single_models(trainX, validX, trainy, validy, testX):
    """
    Run submit functions for single models defined here
    Not SVM because results are too bad
    """
    RF_model = RandomForestClassifier(n_estimators=350, max_features=15, oob_score=True)

    XGB_model = xgb.XGBClassifier(max_depth=7, n_estimators=700, objective='multi:softprob',
                                  learning_rate=0.1, n_jobs=3,
                                  min_child_weight=3, subsample=0.8)

    MLP_model = MLPClassifier((90,60,30), learning_rate_init=0.005, max_iter=50,
                              tol=0.005)

    ET_model = ExtraTreesClassifier(n_estimators=350)

    classif_randomForest(trainX, validX, trainy, validy, testX, RF_model)
    classif_xgboost(trainX, validX, trainy, validy, testX, XGB_model)
    classif_MLP(trainX, validX, trainy, validy, testX, MLP_model)
    classif_ET(trainX, validX, trainy, validy, testX, ET_model)

def averaging_models_by_two(trainX, validX, trainy, validy, testX):
    """
    Run all combinations of 2 models for ensemble averaging
    Best to use averaging_probas to avoid retraining existing models
    """
    averaged_2models(trainX, validX, trainy, validy, testX, RF_model, XGB_model, "RF", "XGB")
    # Best with 0.3 RF 0.7XGB --> 0.447
    averaged_2models(trainX, validX, trainy, validy, testX, RF_model, MLP_model, "RF", "MLP")
    # Best with 0.4 RF 0.6 MLP --> 0.469
    averaged_2models(trainX, validX, trainy, validy, testX, RF_model, ET_model, "RF", "Extra_tree")
    # best 0.2 RF 0.8 Extra_tree --> 0.482
    averaged_2models(trainX, validX, trainy, validy, testX, XGB_model, ET_model, "XGB", "ET")
    # Best with 0.7 XGB et 0.3 ET --> 0.443
    averaged_2models(trainX, validX, trainy, validy, testX, XGB_model, MLP_model, "XGB", "MLP")
    # Best with 0.8 XGB et 0.2 MLP --> 0.45
    averaged_2models(trainX, validX, trainy, validy, testX, MLP_model, ET_model, "MLP", "ET")
    # Best with 0.4 MLP et 0.6 ET --> 0.46

def averaging_probas(validy=[], probvalid=[], numtry=100, probtest=[], weightlist=[], csv_submit="../averaged_probas.csv"):
    """
    Run a stochastic search of good values for ensemble averaging if no weights provided,
    and compute logloss on validation set
    else save probabilities of the result
    Return sorted list of tuple (logloss on validset, list of weights)
    or [] if not corect list of arguments or weightlist and probtest given
    """
    if len(weightlist) and len(probtest):
        weights = np.array(weightlist)[:,np.newaxis, np.newaxis]
        res = np.sum(np.array(probtest) * weights, axis=0)/np.sum(weightlist, axis=0)
        saveResult(res, csv_submit)
        return []
    elif len(validy) and len(probvalid):
        reslist = []
        for i in range(numtry):
            weights = np.random.randint(1,100, len(probvalid))[:,np.newaxis,np.newaxis]
            validres = np.sum(np.array(probvalid) * weights, axis=0)/np.sum(weights, axis=0)
            logloss = evaluation(validy, validres)
            reslist.append((logloss, weights.ravel().tolist()))
        reslist.sort()
        return reslist
    else:
        return []

def main_test():

    X, y, test_X = dex.get_data("../train.csv", "../test.csv") # change to correct path
    train_X, valid_X, train_y, valid_y = dex.data_validation(X, y, split=0.1)
    trainX, trainy = dex.get_preproc_train(train_X, train_y)
    validX, validy = dex.get_preproc_valid(valid_X, valid_y)
    testX = dex.get_preproc_test(test_X)

    run_single_models(trainX, validX, trainy, validy, testX)

    probas_valid_XGB = pd.read_csv('../xgboost_submission_valid.csv').values[:, 1:]
    probas_valid_MLP = pd.read_csv('../MLP_submission_valid.csv').values[:, 1:]
    probas_valid_RF = pd.read_csv('../randomForest_submission_valid.csv').values[:, 1:]
    probas_valid_ET = pd.read_csv('../ET_submission_valid.csv').values[:, 1:]

    probas_valid = [probas_valid_XGB, probas_valid_MLP, probas_valid_RF, probas_valid_ET]

    # Stochastic research 2 models
    results = averaging_probas(validy, probas_valid[:2], 250) 
    print("XGB, MLP", results[:10])
    results = averaging_probas(validy, [probas_valid_XGB,  probas_valid_ET], 250)
    print("XGB, ET", results[:10])
    results = averaging_probas(validy, [probas_valid_XGB, probas_valid_RF], 250)
    print("XGB, RF", results[:10])
    results = averaging_probas(validy, probas_valid[1:3], 250)
    print("MLP, RF", results[:10])
    results = averaging_probas(validy, [probas_valid_MLP, probas_valid_ET], 250)
    print("MLP, ET", results[:10])
    results = averaging_probas(validy, probas_valid[2:], 1000)
    print("ET, RF", results[:10])

    # Stochastic research 3 models
    results = averaging_probas(validy, probas_valid[:3], 1000) 
    print("XGB, MLP, RF", results[:10])
    results = averaging_probas(validy, [probas_valid_XGB, probas_valid_MLP, probas_valid_ET], 1000)
    print("XGB, MLP, ET", results[:10])
    results = averaging_probas(validy, [probas_valid_XGB, probas_valid_RF, probas_valid_ET], 1000)
    print("XGB, RF, ET", results[:10])
    results = averaging_probas(validy, probas_valid[1:], 1000)
    print("MLP, RF, ET", results[:10])


    results = averaging_probas(validy, probas_valid, 1000) # To do stochastic search on validation
    print("All 4 models", results[:10])

def main_submit():
    X, y, test_X = dex.get_data("../train.csv", "../test.csv") # change to correct path
    train_X, valid_X, train_y, valid_y = dex.data_submit(X, y)
    trainX, trainy = dex.get_preproc_train(train_X, train_y)
    testX = dex.get_preproc_test(test_X)

    run_single_models(trainX, valid_X, trainy, valid_y, testX)

    probas_XGB = pd.read_csv('../XGB_submission.csv').values[:, 1:]
    probas_MLP = pd.read_csv('../MLP_submission.csv').values[:, 1:]
    probas_RF = pd.read_csv('../RF_submission.csv').values[:, 1:]
    probas_ET = pd.read_csv('../ET_submission.csv').values[:, 1:]

    probas_test = [probas_XGB, probas_MLP, probas_ET, probas_RF]

    _ = averaging_probas(probtest=probas_test[:2], weightlist=[4, 3], csv_submit="../averaged_XGB_MLP")
    _ = averaging_probas(probtest=[probas_XGB, probas_ET], weightlist=[8, 6], csv_submit="../averaged_XGB_ET")
    _ = averaging_probas(probtest=[probas_XGB, probas_RF], weightlist=[8, 3.5], csv_submit="../averaged_XGB_RF")
    _ = averaging_probas(probtest=[probas_MLP, probas_RF], weightlist=[7.5, 6], csv_submit="../averaged_MLP_RF")
    _ = averaging_probas(probtest=probas_test[1:3], weightlist=[1, 1], csv_submit="../averaged_MLP_ET")
    _ = averaging_probas(probtest=probas_test[2:], weightlist=[2, 9], csv_submit="../averaged_RF_ET")

    _ = averaging_probas(probtest=probas_test[:3], weightlist=[9, 9, 7], csv_submit="../averaged_XGB_MLP_ET") # To save a csv file of averaged probabilities


main_test()
main_submit()
