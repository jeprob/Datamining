"""Skeleton file for your solution to the shortest-path kernel.
Student: Jennifer Probst (16-703-423)"""

import numpy as np

def floyd_warshall(A):
    """Implement the Floyd--Warshall on an adjacency matrix A.

    Parameters
    ----------
    A : `np.array` of shape (n, n)
        Adjacency matrix of an input graph. If A[i, j] is `1`, an edge
        connects nodes `i` and `j`.
        """   
    import copy
    S=copy.deepcopy(A)
    #transform the 0s to 10000 except the diagonal
    n=len(A)
    I = np.identity(n, dtype=bool)
    S[S!=0] = 1 #maybe other distance needed
    S[S==0] = 10000
    S[I]=0
    
    #algorithm as proposed in pseudocode
    for k in range(0,n):
        for i in range(0,n):
            for j in range(0,n):
                if S[i,k] + S[k,j] < S[i,j]:
                    S[i,j]=S[i,k]+S[k,j]

    """
    Returns
    -------
    An `np.array` of shape (n, n), corresponding to the shortest-path
    matrix obtained from A.
    """
    return S

def sp_kernel(S1, S2):
    """Calculate shortest-path kernel from two shortest-path matrices.

    Parameters
    ----------
    S1: `np.array` of shape (n, n)
        Shortest-path matrix of the first input graph.

    S2: `np.array` of shape (m, m)
        Shortest-path matrix of the second input graph.
        """
    sim=0.0
    n=len(S1)
    m=len(S2)
    for ir in range(0, n):
        for ic in range(ir,n):
            for jr in range(0,m):
                for jc in range(jr, m):
                    if S1[ir,ic]==S2[jr,jc] and S1[ir,ic]!=0:
                        sim+=1
    
    """
    Returns
    -------
    A single `float`, corresponding to the kernel value of the two
    shortest-path matrices
    """
    return sim
