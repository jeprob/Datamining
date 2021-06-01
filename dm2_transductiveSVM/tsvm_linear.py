"""
Homework 6: Transductive Support Vector Machines
Course  : Data Mining II (636-0019-00L)

Transductive linear SVM.
This file implements Transductive linear SVM described in the following paper:

Thorsten Joachims. Transductive Inference for Text Classification using Support Vector Machines. ICML 1999.
"""

import svm_linear
import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.metrics.pairwise import linear_kernel


'''
Train linear transductive SVM
Input: 
        X1, matrix of size #sample1 x #feature
        y1, vector of size #sample1, either -1 or 1
        X2, matrix of size #sample2 x #feature
        C1, regularizer parameter, control the slack variable of samples from X1
        C2, regularizer parameter, control the slack variable of samples from X2
        p,  percentage of positive samples in unlabeled data X2, [0, 1]
Output:
        w, weight vector of size #feature
        b, intercept, a float
'''
def train(X1, y1, X2, C1, C2, p):

    # 1. Get number of positive samples
    num_pos = int(X2.shape[0] * p)

    # 2. Train standard SVM using labeled samples
    w, b = svm_linear.train(X1, y1, C1)

    # 3. The num_pos test examples from X2 with highest value of w*x+b are assigned to 1
    # The rest of examples from X2 are assigned to -1
    pred_y_X2 = (np.dot(X2, w) + b)
    indices = np.argsort(pred_y_X2)[-num_pos:]
    y2 = np.ones(pred_y_X2.shape[0])*(-1)
    y2[indices] = 1

    # 4. Retrain with label switchIng
    C_neg = 1e-5
    C_pos = 1e-5 * num_pos / (X2.shape[0] - num_pos)
    while C_neg < C2 or C_pos < C2:
        # 5. Retrain the variant of SVM
        w, b, e1, e2 = svm_linear.train_variant(X1, y1, X2, y2, C1=C1, C2=C_pos, C3=C_neg)
        # 6. Take a positive and negative example, switch their labels
        for m in range(X2.shape[0]):
            for l in range(X2.shape[0]):
                if(y2[m]*y2[l]<0 and e2[l]>0 and e2[m]>0 and e2[m]+e2[l]>2):
                    exists=True #ta
                    y2[m] = -1 * y2[m]
                    y2[l] = -1 * y2[l]
                    w, b, e1, e2 = svm_linear.train_variant(X1, y1, X2, y2, C1=C1, C2=C_pos, C3=C_neg)
        # 7. Increase the value of C_neg and C_pos
        C_neg = min(2*C_neg, C2)
        C_pos = min(2*C_pos, C2)
    # 8. Return the learned model
    return (w, b)