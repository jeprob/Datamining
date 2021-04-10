#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
import math
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


def split_data(X, y, attribute_index, theta):
    '''split X and y in two subsets'''
    a = attribute_index
    X_u = X[X[:,a] >= theta,:]
    X_l = X[X[:,a] < theta,:]
    y_u = y[X[:,a] >= theta]
    y_l = y[X[:,a] < theta]
    
    return X_u, y_u, X_l, y_l


def compute_information_content(X_sub, y):
    ''' compute the information content of a subset X with labels y'''
    
    ic=0
    for i in np.unique(y):
        p=sum(y==i)/len(X_sub)
        ic+=p*math.log2(p)
        
    return -ic


def compute_information_a(X, y, attribute_index, theta):
    '''comp Infoa(X) for dataset X with labels y split accordingly to the split defined'''
    
    X_u, y_u, X_l, y_l = split_data(X, y, attribute_index, theta)
    info = len(X_u)/len(X)*compute_information_content(X_u, y_u) + len(X_l)/len(X)*compute_information_content(X_l, y_l)
    
    return info


def compute_information_gain(X, y, attribute_index, theta):
    ''' Input: X, the Iris data (matrix); 
        y, the Iris label vector;
        attribute_index, the attribute/column index;
        theta, the split value for the attribute indexed by attribute_index.'''
        
    gain = compute_information_content(X, y) - compute_information_a(X, y, attribute_index, theta)
    
    return gain


if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))
    target_names = iris.target_names

    ####################################################################
    print('------------\n')
    print('Exercise 2.b')

    split1 = compute_information_gain(X, y, 0, 5.0)
    split2 = compute_information_gain(X, y, 1, 3.0)
    split3 = compute_information_gain(X, y, 2, 2.5)
    split4 = compute_information_gain(X, y, 3, 1.5)

    print('Split ( sepal length (cm) < 5.0): information gain = {},'.format(round(split1,2)))
    print('Split ( sepal width (cm) < 3.0): information gain = {},'.format(round(split2,2)))
    print('Split ( petal length (cm) < 2.5): information gain = {},'.format(round(split3,2)))
    print('Split ( petal witdh (cm) < 1.5): information gain = {},'.format(round(split4,2)))

    
    print('------------\n')
    print('Exercise 2.c')
    print('We want to pick the split such that the gain is maximized (and equivalently the entropy reduced), so I would pick the spit 3: petal length < 2.5, displaying the highest gain value.')
    print('------------\n')
    print('Exercise 2.d')
    np.random.seed(42)
    
    cv = KFold(n_splits=5, shuffle=True)
    accuracies = []
    feature_imp = []
    
    for train_index, test_index in cv.split(X, y): #train tree on each fold
        #split data
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        #classifier
        clf = DecisionTreeClassifier()
        #fit on train data
        clf.fit(X_train, y_train)
        #predict on test data
        y_pred = clf.predict(X_test)
        #get accuracy
        accuracies.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        feature_imp.append(list(clf.feature_importances_))
        
    print('Accuracy score using cross-validation is : {}'.format(round(np.mean(accuracies)*100,2)))
    
    print('-------------------------------------\n')
    print('Feature importances for _original_ data set')
    #attribute of each classifier in each fold
    print(feature_imp)
    print('For the original data, the two most important features are: petal width (fourth value for each feature importance vector) having by far the highest feature importance and petal length (third value) displaying the second highest values.')
    
    print('-------------------------------------------\n')
    print('Feature importances for _reduced_ data set')
    #remove all samples with y==2
    X = X[y != 2]
    y = y[y != 2]
    #re-run crossvalidation
    cv2 = KFold(n_splits=5, shuffle=True)
    feature_imp = []
    
    for train_index, test_index in cv.split(X, y): #train tree on each fold
        #split data
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        #classifier
        clf = DecisionTreeClassifier()
        #fit on train data
        clf.fit(X_train, y_train)
        #predict on test data
        y_pred = clf.predict(X_test)
        #get accuracy
        feature_imp.append(list(clf.feature_importances_))
    
    print(feature_imp)
    
    print('For the reduced data, the most important feature is petal length, being the only important feature in all crossvalidation runs (score of 1.0).')
    print('This means that petal width is an important feature to separate virginica from setosa and versicolor. But the later two seem to be distinguishable only by petal length.')
    print('------------------------------------------\n')
