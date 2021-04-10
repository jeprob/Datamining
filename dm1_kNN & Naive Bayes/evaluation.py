"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the metrics that are invoked from the main program.

Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    '''
    Function for calculating TP, FP, TN, and FN.
    The input includes the vector of true labels
    and the vector of predicted labels
    '''
    
    N = np.size(y_true)
    same = (y_true == y_pred)
    pos_pred = (y_pred=='+')
    neg_pred = (y_pred=='-')
    TP = sum(pos_pred[pos_pred==same])
    FP = sum(pos_pred)-TP
    TN = sum(neg_pred[neg_pred==same])
    FN = sum(neg_pred)-TN
    
    conf = np.array([TP/N, FP/N, TN/N, FN/N])
    
    '''
    output: TP, FP, TN, and FN values
    '''
    return conf


def compute_precision(y_true, y_pred):
    """
    Function: compute_precision
    Invoke confusion_matrix() to obtain the counts
    """

    conf1 = confusion_matrix(y_true, y_pred)
    precision = conf1[0]/(conf1[0]+conf1[1])

    return precision


def compute_recall(y_true, y_pred):
    """
    Function: compute_recall
    Invoke confusion_matrix() to obtain the counts
    """
    
    conf2 = confusion_matrix(y_true, y_pred)
    recall = conf2[0]/(conf2[0]+conf2[3])
    
    return recall


def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Invoke the confusion_matrix() to obtain the counts
    """
    
    conf3 = confusion_matrix(y_true, y_pred)
    accuracy = conf3[0]+conf3[2]/sum(conf3)
    
    return accuracy

#%%

y_true = np.array(['-','-','-','+'])
y_pred = np.array(['-','-','+','+'])
conf = confusion_matrix(y_true, y_pred)
prec = compute_precision(y_true, y_pred)
rec = compute_recall(y_true, y_pred)
acc = compute_accuracy(y_true, y_pred)