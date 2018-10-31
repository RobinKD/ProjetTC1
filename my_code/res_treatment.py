import numpy as np
import pandas as pd
import data_extraction as dex

def evaluation(label,pred_label):
    keep_numclass = np.vectorize(lambda w: int(w[-1]))
    labels = keep_numclass(label)
    num = len(labels)
    num_labels = len(pred_label[0])
    logloss = 0.0
#     print(labels)
#     print(pred_label)
    for i in range(num):
        p = max(min(pred_label[i][labels[i]-1],1-10**(-15)),10**(-15))
        logloss += np.log(p)
    logloss = -1*logloss/num
    return logloss

def saveResult(probas, filename = "../submission.csv"):
#     col = np.concatenate((['id'], np.unique(panda_dataset['target'].values)))
    col = np.unique(dex.panda_dataset['target'].values)
    indices = np.arange(len(probas)).reshape(-1,1)
#     submit = np.concatenate((indices, probas), axis=1)
    submission = pd.DataFrame(probas, columns=col)
    submission.index += 1
#     print(submission)
    submission.to_csv(filename, index=True, index_label='id')


# prediction = model.predict_proba(test)
# saveResult(prediction)
