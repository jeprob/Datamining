import numpy as np
import numpy.linalg as linalg
import scipy as sp

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
Input: X: matrix to invert
       tol: tolerance cut-off to exclude tiny singular values (default=1e15)
Output: Pseudo-inverse of X.
Note: Do not use scipy or numpy pinv method. Implement the function yourself.
      You can of course add an assert to compare the output of scipy.pinv to your implementation
'''
def compute_pinv(X=None,tol=1e-15):
    L, s, R = sp.linalg.svd(X)
    s[s>tol] = 1/s
    st = np.diag(np.full(len(s),s))
    zeroos = np.zeros((np.shape(X)[1], np.shape(X)[0]-np.shape(st)[0]))
    stfin = np.concatenate((st, zeroos), axis=1)
    return R.T @ stfin @ L.T