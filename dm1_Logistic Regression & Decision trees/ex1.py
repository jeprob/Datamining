'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":

    ###################################################################
    # Your code goes here.
    
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="kNN of microRNA expression set of breast cancer"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to directory containing train data"
    )
    parser.add_argument(
        "--testdir",
        required=True,
        help="Path to directory containing test data"
    )
    
    args = parser.parse_args()
    
    #set directories:
    trainpath = args.traindir
    testpath = args.testdir
    
    #read data from file using pandas
    train = pd.read_csv("{}/{}".format(trainpath, "diabetes_train.csv"))
    test = pd.read_csv("{}/{}".format(testpath, "diabetes_test.csv"))
    
    # extract first 7 columns to data matrix X (actually, a numpy ndarray)
    X_train = train.iloc[:, 0:7].values
    X_test = test.iloc[:, 0:7].values

    # extract 8th column (labels) to numpy array
    y_train = train.iloc[:, 7].values
    y_test = test.iloc[:, 7].values
    
    #scale
    scaler = StandardScaler()
    scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #training
    clf = LogisticRegression(C=1, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    #prediction
    y_pred = clf.predict(X_test)
    
    ###################################################################
    #PRINTS
    print('Exercise 1.a')
    compute_metrics(y_test, y_pred)
    print('------------')
    print('Exercise 1.b')
    print('Both classifiers roughly classify the same number of samples correct (TP+TN) in the diabetes dataset. '
          'The logistic regression generally predicts less patients to have diabetes (positive prediction). ' 
          'I think it would be better if more patients would be classified as positive and then further checks could be made, therefore I would choose the LDA method.')
    print('------------')
    ###################################################################
    print('Exercise 1.c')
    print('I think I would choose Logistic Regression as it makes no assumptions on the distribution of the explanatory data.'
          'LDA was developed for normally distributed explanatory variables, therefore it might have predicted better on our data, also taking covariance into acount which might have been in the measured attributes.'
          'I expect Logistic Regression in all other situations if normality is not fulfulled to perform better.'
          'Additionally Logistic Regression also has a better runtime for large datasets.')
    ###################################################################
    print('-------------')
    print('Exercise 1.d')
    print('predicted coefficients:', clf.coef_)
    print('The two attributes which appear to contribute the most to the prediction are glu and ped. We find this by looking at the predicted coefficients, which is highest for glu and ped (0.97 and 0.53).')
    print('The coefficient for age is 0.435.')
    print('Calculating the exponential function results in', round(np.exp(0.434952817),4), 'which amounts to an increase in diabetes risk of 1.5449/(1.5449+1) =  60.71 percent per additional year.')
    print('Performance on the reduced dataset:')
    
    train_reduced = train.drop(['skin'], axis=1)
    test_reduced = test.drop(['skin'], axis=1)
    X_train_reduced = train_reduced.iloc[:, 0:6].values
    X_test_reduced = test_reduced.iloc[:, 0:6].values
    y_train_reduced = train_reduced.iloc[:, 6].values
    y_test_reduced = test_reduced.iloc[:, 6].values
    
    scaler2 = StandardScaler()
    X_train_reduced = scaler2.fit_transform(X_train_reduced, y_train_reduced)
    X_test_reduced = scaler2.transform(X_test_reduced)
    
    clf2 = LogisticRegression(C=1, max_iter=150, solver='lbfgs')
    clf2.fit(X_train_reduced, y_train_reduced)
    
    #prediction
    y_pred2 = clf2.predict(X_test_reduced)

    compute_metrics(y_test_reduced, y_pred2)
    
    print('By comparing the performance and the coefficients obtained on the reduced dataset with the ones on the model including all the attributes, I observe that the output stays exactly the same. My explanation is that there is no or barely any influence of scin on diabetes.'
          'We can also see this in the results of the logistic regression as skin has a coefficient of 0.0007.')
    print('------------')
    #####################################################################
    
    
