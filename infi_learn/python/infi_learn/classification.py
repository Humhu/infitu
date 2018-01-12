"""Classes for using X to predict success
"""
import math
import numpy as np
import scipy.stats as ss
import scipy.spatial.distance as ssd
import scipy.integrate as si

from itertools import izip

def compute_auc(tprs, fprs):
    # NOTE Can be negative if fprs are sorted descending
    return abs(si.trapz(y=tprs, x=fprs))

def compute_threshold_roc(probs, labels, n=30):
    """Computes a receiver operating characteristic (ROC) curve displaying true positive rate
    versus false positive rate over a range of probability thresholds. 
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels, dtype=bool)

    thresholds = np.linspace(0, 1.0, num=n)
    num_positives = float(np.sum(labels))
    num_negatives = len(labels) - num_positives
    tprs = []
    fprs = []
    for t in thresholds:
        preds = probs > t
        true_pos = np.logical_and(preds, labels)
        false_pos = np.logical_and(preds, np.logical_not(labels))
        tpr = np.sum(true_pos) / num_positives
        fpr = np.sum(false_pos) / num_negatives
        tprs.append(tpr)
        fprs.append(fpr)
    return tprs, fprs



def compute_classification_loss(classifier, x, y, mode='logistic', balance=True, params=None):
    """Computes the classification loss of a classifier with the given
    log hyperparameters on a dataset (x,y)
    """
    if params is not None:
        classifier.log_params = params

    if len(x) == 0:
        return 0

    x = np.asarray(x)
    classes = np.asarray(y, dtype=bool)
    if balance:
        pos_loss = compute_classification_loss(classifier,
                                                 x=x[classes],
                                                 y=classes[classes],
                                                 mode=mode,
                                                 balance=False,
                                                 params=params)
        n_classes = np.logical_not(classes)
        neg_loss = compute_classification_loss(classifier,
                                                 x=x[n_classes],
                                                 y=classes[n_classes],
                                                 mode=mode,
                                                 balance=False,
                                                 params=params)
        return (pos_loss + neg_loss) * 0.5

    y = np.asarray(y, dtype=float) * 2 - 1.0
    pp = classifier.query(x) * 2 - 1.0

    if mode == 'mse':
        err = y - pp
        return np.mean(err * err)
    elif mode == 'logistic':
        return np.mean(np.log(1 + np.exp(-y * pp))) / math.log(2)
    else:
        raise ValueError('Unrecognized classification loss: %s' % mode)

