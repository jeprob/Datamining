# -*- coding: utf-8 -*-
"""
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
# Author: Jennifer Probst, Mai 2021

from somutils import *
import os
import argparse
import sys
import pandas as pd
import numpy as np


if __name__ == '__main__':
    
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="SOM Exercise 1 and 2"
    )
    parser.add_argument(
        "--exercise",
        required=True, type=int,
        help= "either 1 or 2 indicating what exercise the program will solve"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help= "Path to the output directory"
    )
    parser.add_argument(
        "--p", type=int,
        required=True,
        help="Number of rows in the grid"
    )
    parser.add_argument(
        "--q", type=int,
        required=True,
        help="Number of columns in the grid"
    )
    parser.add_argument(
        "--N", type=int,
        required=True,
        help="Number of iterations"
    )
    parser.add_argument(
        "--alpha_max", type=int,
        required=True,
        help="Upper limit for learning rate"
    )
    parser.add_argument(
        "--epsilon_max", type=int,
        required=True,
        help="Upper limit for radius"
    )
    parser.add_argument(
        "--lamb", type=float,
        required=True,
        help="Decay constant (lambda) for learning rate decay"
    )
    parser.add_argument(
        "--file",
        required=False,
        help="Full path to the input file in Exercise 2"
    )
    
    args = parser.parse_args()
    
    # path setup
    data_dir = args.file
    out_dir = args.outdir
    os.makedirs(args.outdir, exist_ok=True)
    
    
    
    #in case of exercise 1
    if args.exercise==1:
        scurve = makeSCurve()
        buttons, grid, error = SOM(scurve, args.p, args.q, args.N, args.alpha_max, args.epsilon_max, True, args.lamb)
        path_1b = args.outdir + "/exercise_1b.pdf"
        path_1c = args.outdir + "/exercise_1c.pdf"
        plotDataAndSOM(scurve, buttons, path_1b)
        plotReconstructionError(error,  path_1c)
        
        
    #in case of exercise 2
    
    if args.exercise==2:
        #exercise 2a:
        # Create the output file
        try:
            file_name = "{}/output_some_crabs.txt".format(args.outdir)
            f_out = open(file_name, 'w')
        except IOError:
            print("Output file {} cannot be created".format(file_name))
            sys.exit(1)
        
        # Write header for output file
        f_out.write('{}\t{}\t{}\t{}\n'.format('sp', 'sex', 'index', 'label'))
        
        # Load input file and compute buttons
        crabdata = pd.read_csv(args.file)
        info = crabdata.iloc[:,0:3]
        info = info.to_numpy()
        features = crabdata.iloc[:,3:8]
        features = features.to_numpy()
        buttons, grid, error = SOM(features, args.p, args.q, args.N, args.alpha_max, args.epsilon_max, False, args.lamb)

        for i in range(info.shape[0]):
            nearest = findNearestButtonIndex(features[i,:], buttons)
            f_out.write('{}\t{}\t{}\t{}\n'.format(info[i,0], info[i,1], info[i,2], nearest))

        #Close the file
        f_out.close()



        #exercise 2b
        path_2b = args.outdir + "exercise_2b.pdf"
        plotSOMCrabs(features, info, grid, buttons, "exercise_2b.pdf")