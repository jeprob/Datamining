#import all necessary functions
from utils import *
from pca import *
from pinv import *
import numpy as numpy

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:

    #Get Iris Data
    data = loadIrisData()
    #0. normalise data
#   data.data = dataNormalisation(data.data)
    
    #Perform a PCA using covariance matrix and eigen-value decomposition
    #1. Compute covariance matrix
    cov_mat = computeCov(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    values_pca, vectors_pca = computePCA(cov_mat)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    t_data_pca = transformData(vectors_pca[:,0:2], data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(t_data_pca, data.target, filename="pca_plot_transformeddata.pdf")
    #5. How much variance can be explained with each principle component?
    var_pca = computeVarianceExplained(values_pca)
    np.set_printoptions(precision=2)
    print("Variance Explained PCA: ")
    for i in range(var_pca.shape[0]):
        print("PC %d: %.2f"%(i+1,var_pca[i]))
    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(var_pca,filename="cumvar_pca.pdf")


    #Perform a PCA using SVD
    #1. Normalise data by substracting the mean
    datam = zeroMean(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    values_svd, vectors_svd = computePCA_SVD(datam)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    t_data_svd = transformData(vectors_svd[:,0:2], datam)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(t_data_svd, data.target, filename="svd_plot_transformeddata.pdf")
    #5. How much variance can be explained with each principle component?
    var_svd = computeVarianceExplained(values_svd)
    np.set_printoptions(precision=2)
    print("Variance Explained PCA: ")
    for i in range(var_svd.shape[0]):
        print("PC %d: %.2f"%(i+1,var_svd[i]))
    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(var_svd,filename="cumvar_svd.pdf")


    #Exercise 3
    #1. Compute the Moore-Penrose Pseudo-Inverse on the Iris data
    pseudoinv = compute_pinv(data.data)
    #2. Check Properties

    print("\nChecking status exercise 3:")
    status = (data.data @ pseudoinv @ data.data).all() == data.data.all()
    print(f"X X^+ X = X is {status}")
    status = (pseudoinv @ data.data @ pseudoinv).all() == pseudoinv.all()
    print(f"X^+ X X^+ = X^+ is {status}")
