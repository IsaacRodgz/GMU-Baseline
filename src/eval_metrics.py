import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def metrics(results, truths):
    preds = results.cpu().detach().numpy()
    truth = truths.cpu().detach().numpy()
    '''
    print(preds.shape)
    print(truth.shape)
    print("------------")
    print(preds[0])
    print(truth[0])
    input()
    '''
    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    f_score_micro = f1_score(truth, preds, average='micro', zero_division=0)
    f_score_macro = f1_score(truth, preds, average='macro', zero_division=0)
    f_score_weighted = f1_score(truth, preds, average='weighted', zero_division=0)
    f_score_samples = f1_score(truth, preds, average='samples', zero_division=0)
    accuarcy = accuracy_score(truth, preds)

    return accuarcy, f_score_micro, f_score_macro, f_score_weighted, f_score_samples


def multiclass_acc(results, truths):
    preds = results.view(-1).cpu().detach().numpy()
    truth = truths.view(-1).cpu().detach().numpy()
    
    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    return np.sum(preds == truths) / float(len(truths))