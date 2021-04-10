"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy as np
import math

def manhattan_dist(v1, v2):
    
    #compute distance:
    dist=sum(abs(v1-v2))

    return dist

def hamming_dist(v1, v2):
    
    #binary vectors
    v1new=np.zeros(len(v1))
    v2new=np.zeros(len(v2))
    v1new[v1 > 0] = 1
    v2new[v2 > 0] = 1
        
    #compute distance:
    dist=sum(abs(v1new-v2new))    
    
    return dist  

def euclidean_dist(v1, v2):
    
    #compute distance: 
    dist= (sum((v1-v2)**2))**(0.5)
    
    return dist

def chebyshev_dist(v1, v2):
    
    #compute distance:
    dist = max(abs(v1-v2))
    
    return dist 

def minkowski_dist(v1, v2, d):
    
    #compute distance: 
    dist = sum((abs(v1-v2))**d)**(1/d)
    
    return dist 