"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""

#import all necessary functions
from utils import *
from pca import *
import scipy as sp
import numpy as np

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:
    
    #Simulate Data
    data = simulateData()
    #Perform a PCA
    #1. Compute covariance matrix
    covariance_matrix = computeCov(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    PCA = computePCA(covariance_matrix)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    pcs_to_use = PCA[1:,0:2]
    t_data = transformData(pcs_to_use, data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(t_data, data.target, filename="exercise_2_1.pdf")
    #5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(PCA[0,:])
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var[i]))
    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(var,filename="cumvar_2_1.pdf")
    
    
    ##################
    #Exercise 2 Part 2:
    
    #1. normalise data
    norm_data = dataNormalisation(data.data)
    #2. compute covariance matrix
    covariance_matrix2 = computeCov(norm_data)
    #3. compute PCA
    PCA2 = computePCA(covariance_matrix2)
    #4. Transform your input data inot a 2-dimensional subspace using the first two PCs
    pcs_to_use2 = PCA2[1:,0:2]
    t_data2 = transformData(pcs_to_use2, norm_data)
    #5. Plot your transformed data
    plotTransformedData(t_data2, data.target, filename="exercise_2_2.pdf")
    #6. Compute Variance Explained
    var2 = computeVarianceExplained(PCA2[0,:])
    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.2: ")
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var2[i]))
    #7. Plot Cumulative Variance
    plotCumSumVariance(var2,filename="cumvar_2_2.pdf")