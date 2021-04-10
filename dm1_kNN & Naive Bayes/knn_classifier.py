import os
import numpy as np
from scipy.spatial.distance import cdist


class KNNClassifier:
    '''
    A class object that implements the methods of a k-Nearest Neighbor classifier
    The class assumes there are only two labels, namely POS and NEG

    Attributes of the class
    -----------------------
    k : Number of neighbors
    X : A matrix containing the data points (train set)
    y : A vector with the labels
    dist : Distance metric used. Possible values are: 'euclidean', 'hamming', 'minkowski', and others
           For a full list of possible metrics have a look at:
           http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    HINT: for using the attributes of the class in the class' methods, you can use: self.attribute_name
          (e.g. self.X for accessing the value of the attribute named X)
    '''     
    def __init__(self, X, y, metric):
        '''
        Constructor when X and Y are given.
        
        Parameters
        ----------
        X : Matrix with data points
        Y : Vector with class labels
        metric : Name of the distance metric to use
        '''
        # Default values
        self.verbose = False
        self.k = 1

        # Parameters
        self.X = X
        self.y = y
        self.metric = metric


    def debug(self, switch):
        '''
        Method to set the debug mode.
        
        Parameters
        ----------
        switch : String with value 'on' or 'off'
        '''
        self.verbose = True if switch == "on" else False


    def set_k(self, k):
        '''
        Method to set the value of k.
        
        Parameters
        ----------
        k : Number of nearest neighbors
        '''
        self.k = k


    def _compute_distances(self, X, x):
        '''
        Private function to compute distances. 
        Compute the distance between x and all points in X
    
        Parameters
        ----------
        x : a vector (data point)
        '''
        
        x = np.array([x])
        xm = np.repeat(x, repeats = len(self.X), axis=0)
        return cdist(self.X, xm, self.metric)[:,0]
        

    def predict(self, x):
        '''
        Method to predict the label of one data point.
        Here you actually code the KNN algorithm.
       
        Hint: for calling the method _compute_distance 
              (which is private), you can use: 
              self._compute_distances(self.X, x) 
        
        Parameters
        ----------
        x : Vector from the test data.
        '''
        #alle selfs noch raus bei distances and nearesty
        #get list of distances
        distances = self._compute_distances(self.X, x)
        N = len(distances)
        
        #get y from k nearest x
        nearest_y = []
        nearest_d = np.empty([self.k, 1])
        for i in range(N):
            r = distances[i]
            if i < self.k:
                nearest_y.append(self.y[i])
                nearest_d[i]=r
            elif r<max(nearest_d): #find the ones with lowest distance
                a=nearest_d.argmax()
                nearest_d[a]=r
                nearest_y[a]=self.y[i]
                
        nearest_y=np.array(nearest_y)
        
        #get ave lable: 
        pos = sum(nearest_y=='+')
        neg= sum(nearest_y=='-')
        
        if pos==neg:
            return 0
        elif pos > neg: 
            return('+')
        elif neg > pos: 
            return('-')
        
