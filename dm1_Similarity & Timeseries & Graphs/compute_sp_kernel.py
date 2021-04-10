# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:19:57 2020
Student: Jennifer Probst (16-703-423)
"""
import numpy as np
import os
import argparse

from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import sp_kernel
    

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="SP-Kernel")
parser.add_argument("--datadir", required=True)
parser.add_argument("--outdir", required=True)

args = parser.parse_args()

# Set the paths
data_dir = args.datadir
out_dir = args.outdir

os.makedirs(args.outdir, exist_ok=True)

#import adjacency matrices
import scipy.io
mat = scipy.io.loadmat(r"{}/{}".format(args.datadir, 'MUTAG.mat'))
label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
data = np.reshape(mat['MUTAG']['am'], (len(label), ))


# Create the output file
file_name = "{}/SP_output.txt".format(args.outdir)
f_out = open(file_name, 'w')
f_out.write('{}\t{}\n'.format('Pair of classes', ' SP')) # Write header
 
cdict = {}
cdict['mutagenic'] = 1
cdict['non-mutagenic'] = -1
groups = ['mutagenic', 'non-mutagenic']

for ind_1 in range(len(groups)):
    for ind_2 in range(len(groups)):
        #get data
        group1 = data[label == cdict[groups[ind_1]]]
        group2 = data[label == cdict[groups[ind_2]]]
        #average similarities
        count = 0
        total = 0
        for mat_1 in group1:
            for mat_2 in group2:
                #get matrices
                trans_1 = floyd_warshall(mat_1)
                trans_2 = floyd_warshall(mat_2)
                #compute distance
                total += sp_kernel(trans_1, trans_2)
                count += 1
        ave = round(total/count, 2)
        # Save the output
        f_out.write('{}:{}\t{}\n'.format(groups[ind_1], groups[ind_2], ave))
        
f_out.close()