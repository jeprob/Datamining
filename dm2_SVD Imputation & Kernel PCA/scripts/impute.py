"""
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import copy
from sklearn.metrics import mean_squared_error
import numpy as np

'''
Impute missing values using the mean of each feature
Input: X: data matrix with missing values (sp.nan) of size n x m, 
          where n is the number of samples and m the number of features
Output: D_imputed (n x m): Mean imputed data matrix
'''
def mean_imputation(X=None):
    D_imputed = X.copy()
    #Impute each missing entry per feature with the mean of each feature
    for i in range(X.shape[1]):
        feature = X[:,i]
        #get indices for all non-nan values
        indices = sp.where(~sp.isnan(feature))[0]
        #compute mean for given feature
        mean = sp.mean(feature[indices])
        #get nan indices
        nan_indices = sp.where(sp.isnan(feature))[0]
        #Update all nan values with the mean of each feature
        D_imputed[nan_indices,i] = mean
    return D_imputed

'''
Impute missing values using SVD
Input: X: data matrix with missing values (sp.nan) of size n x m,
          where n is the number of sampkes and m the number of features
       rank: rank for the rank-r approximation of the orginal data matrix
       tol: precision tolerance for iterative optimisier to stop (default=0.1). The smaller the better!
       max_iter: maximum number of iterations for optimiser (default=100)
Output: D_imputed (n x m): Mean imputed data matrix
'''
def svd_imputation(X=None,rank=None,tol=.1,max_iter=100):
    #get all nan indices
    nan_indices = sp.where(sp.isnan(X))
    #initialise all nan entries with the a mean imputation
    D_imputed = mean_imputation(X)
    #repeat approximation step until convergance
    for i in range(max_iter):
        D_old = D_imputed.copy()
        #SVD on mean_imputed data
        [L,d,R] = linalg.svd(D_imputed)
        #compute rank r approximation of D_imputed
        D_r = sp.matrix(L[:,:rank])*sp.diag(d[:rank])*sp.matrix(R[:rank,:])
        #update imputed entries according to the rank-r approximation
        imputed = D_r[nan_indices[0],nan_indices[1]]
        D_imputed[nan_indices[0],nan_indices[1]] = sp.asarray(imputed)[0]
        #use Frobenius Norm to compute similarity between new matrix and the latter approximation
        fnorm = linalg.norm(D_old-D_imputed,ord="fro")
        if fnorm<tol:
            print("\t\t\t[SVD Imputation]: Converged after %d iterations"%(i+1))
            break
        if (i+1)>=max_iter:
            print("\t\t\t[SVD Imputation]: Maximum number of iterations reached (%d)"%(i+1))
    return D_imputed

'''
Find Optimal Rank-r Imputation
Input: X: data matrix with missing values (sp.nan) of size n x m,
          where n is the number of samples and m the number of features
       ranks: int array with r values to use for optimisation
       test_size: float between 0.0 and 1.0 and represent the proportion of the
                  non-nan values of the data matrix to use for optimising the rank r
                  (default: 0.25)
       random_state: Pseudo-random number generator state used for random splitting (default=0)
       return_optimal_rank: return optimal r (default: True)
       return_errors: return array of testing-errors (default: True)
Output: X_imputed: imputed data matrix using the optimised rank r
        r: optimal rank r [if flag is set]
        errors: array of testing-errors [if flag is set]
'''
def svd_imputation_optimised(X=None,ranks=None,
                             test_size=None,random_state=0,
                             return_optimal_rank=True,return_errors=True):
    # init variables
    sp.random.seed(random_state)
    testing_errors = []
    optimal_rank = sp.nan
    minimal_error = sp.inf

    # 1. Get all non-nan indices
    non_nan_indices = sp.where(~sp.isnan(X))
    non_nan_indices_rows = non_nan_indices[0]
    non_nan_indices_cols = non_nan_indices[1]

    # 2. Use "test_size" % of the non-missing entries as test data
    test_train_array = np.arange(len(non_nan_indices_rows))
    train_indices = np.random.choice(test_train_array, size=int(test_size*len(non_nan_indices_rows)), replace=False)
    train_indices_rows = non_nan_indices_rows[train_indices]
    train_indices_cols = non_nan_indices_cols[train_indices]

    # 3. Create a new training data matrix
    X_train = X.copy()
    X_train[train_indices_rows, train_indices_cols] = np.nan

    # 4.Find optimal rank r by minimising the Frobenius-Normusing the train and test data for rankin ranks:
    for rank in ranks:
        print("\t Testing rank %d ..." %(rank))
        # 4.1 Impute Training Data
        X_imp = svd_imputation(X_train, rank)
        # 4.2 Compute the mean squared error of imputed test data with original test data and store error in array
        error = mean_squared_error(X[train_indices_rows, train_indices_cols], X_imp[train_indices_rows, train_indices_cols])
        testing_errors.append(error)
        print("\t\tMeanSquaredError:%.2f" % error)

        # 4.3 Update rank if necessary
        if error<minimal_error:
            minimal_error=error
            optimal_rank=rank
            X_imputed = X_imp

    # 5. Use optimal rank for imputing the "original" data matrix
    print("Optimal Rank: %f(Mean-SquaredError: %.2f)" % (optimal_rank, minimal_error))

    # return data
    if return_optimal_rank == True and return_errors == True:
        return [X_imputed, optimal_rank, testing_errors]
    elif return_optimal_rank == True and return_errors == False:
        return [X_imputed, optimal_rank]
    elif return_optimal_rank == False and return_errors == True:
        return [X_imputed, testing_errors]
    else:
        return X_imputed