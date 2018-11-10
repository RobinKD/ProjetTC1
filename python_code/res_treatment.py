import numpy as np
import pandas as pd
import data_extraction as dex

def evaluation(label,pred_label):
    """
    Implementation of the evaluation function provided
    by the challenge on kaggle
    """
    num = len(label)
    num_labels = len(pred_label[0])
    logloss = 0.0
    for i in range(num):
        p = max(min(pred_label[i][label[i]],1-10**(-15)),10**(-15))
        p /= np.sum(pred_label[i])
        logloss += np.log(p)
    logloss = -logloss/num
    return logloss

def saveResult(probas, filename = "../submission.csv"):
    """
    Writing the probabilities computed in a csv file having
    the correct form for submission
    """
    col = np.unique(pd.read_csv("../train.csv")['target'].values)
    submission = pd.DataFrame(probas, columns=col)
    submission.index += 1
    submission.to_csv(filename, index=True, index_label='id')


