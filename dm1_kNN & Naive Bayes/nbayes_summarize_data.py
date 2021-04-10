# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:37:23 2020

@author: probst.jennifer
legi: 16703423
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd



# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="kNN of microRNA expression set of breast cancer"
)
parser.add_argument(
    "--traindir",
    required=True,
    help="Path to directory containing train data"
)

parser.add_argument(
    "--outdir",
    required=True,
    help="Path to directory where output will be created"
)

args = parser.parse_args()



#set arguments:
trainpath = args.traindir



# If the output directory does not exist, then create it
out_dir = args.outdir
os.makedirs(args.outdir, exist_ok=True)
  


#Read the training and test data. For each dataset, get also the true labels. 
train = pd.read_table("{}/{}".format(trainpath, "tumor_info.txt"), sep="	", names=['clump','uniformity', 'marginal', 'mitoses', 'Value'])



#calculate the probabilites and write in files
for i in np.array([2,4]):
    train_subs = train[train['Value'] == i]
    # Create the output file
    try:
        file_name = "{}/output_summary_class_{}.txt".format(args.outdir,i)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)
        
    # Write header for output file
    f_out.write('{}\t{}\t{}\t{}\t{}\n'.format('Value', 'clump', 'uniformity',
                'marginal', 'mitoses'))
    
    clumpsum=0
    for val in range(1,11):
        clump = sum(train_subs['clump'] == val)/train_subs['clump'].count()
        uniformity = sum(train_subs['uniformity'] == val)/train_subs['uniformity'].count()
        marginal = sum(train_subs['marginal'] == val)/train_subs['marginal'].count()
        mitoses = sum(train_subs['mitoses'] == val)/train_subs['mitoses'].count()
    
        f_out.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(val, clump, uniformity, marginal, mitoses))
            
    #Close the file       
    f_out.close()
    
