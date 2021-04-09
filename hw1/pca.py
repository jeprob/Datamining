import scipy as sp
from statistics import stdev
import pylab as pl
import numpy as np
from numpy.linalg import eig

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    
    n = np.shape(X)[1]
    
    covmat = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            covmat[i,j] = np.dot(X[:,i] - np.mean(X[:,i]), X[:,j] - np.mean(X[:,j])) / (n-1)
            
    return covmat

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    
    values, vectors = eig(matrix)
    inds = values.argsort()[::-1]
    values = values[inds]
    vectors = vectors[:,inds]
    
    l = np.row_stack((values, np.transpose(vectors)))
    
    return l

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
     
    data_compressed = np.matmul(data, pcs)
    
    return data_compressed

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    
    norm = sum(evals)
    var_expl = evals / norm
    
    return (var_expl)


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Transformed Data
Input: transformed: data matrix (#samples x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename="exercise1.pdf"):
    
    pl.figure()
    #PLOT FIGURE HERE
    pl.scatter(transformed[:,0], transformed[:,1], c=labels)
    pl.xlabel('First PC')
    pl.ylabel('Second PC')
    #Save File
    pl.savefig(filename)

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save file
    pl.savefig(filename)



'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Exercise 2 Part 2:
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    
    normdata= np.zeros((np.shape(X)[0],np.shape(X)[1]))
    
    for i in range(np.shape(X)[1]):
        normdata[:,i] = np.divide(np.subtract(X[:,i], np.mean(X[:,i])), np.std(X[:,i]))
    
    return normdata
