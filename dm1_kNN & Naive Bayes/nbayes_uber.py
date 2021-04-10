# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:56:10 2020

@author: probst.jennifer
16-703-423
"""
import numpy as np
import matplotlib.pyplot as plt

#given values
Nmax = 1000
D=60

#likelyhood function depending on N
def likely(N):
    like=1/N
    return like

#known propabilities
prior=1/Nmax

evidence=0
for N in range(D,Nmax+1):
    evidence+=likely(N)*prior


#get the wanted probabilies for values of N from D to Nmax, 0 probabilities for N<D
prob = []
for i in range(Nmax):
    if i<59:
        prob.append(0)
    else: 
        x = (likely(i)*prior)/(evidence)
        prob.append(x)
        
#for which value max prob
print(prob.index(max(prob)))


#plot the posterior probabilities
fig = plt.figure(figsize=(10,5))
plt.plot(prob)
plt.xlabel('Nmax') 
plt.ylabel('posterior probability') 
plt.title('Plot of posterior probabilites depending on Nmax') 
fig.savefig('posteriordistr.jpg')
