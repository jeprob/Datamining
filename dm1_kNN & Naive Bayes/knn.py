"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for k-NN.
Predicts the labels of the test data using the training data.
The k-NN algorithm is executed for different values of k (user-entered parameter)


Original author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import argparse
import os
import sys
import numpy as np

# Import the file with the performance metrics 
import evaluation

# Class imports
from knn_classifier import KNNClassifier


# Constants
# 1. Files with the datapoints and class labels
DATA_FILE  = "matrix_mirna_input.txt"
PHENO_FILE = "phenotype.txt"

# 2. Classification performance metrics to compute
PERF_METRICS = ["accuracy", "precision", "recall"]


def load_data(dir_path): 
    """
    Function for loading the data.
    Receives the path to a directory that will contain the DATA_FILE and PHENO_FILE.
    Loads both files into memory as numpy arrays. Matches the patientId to make
    sure the class labels are correctly assigned.

    Returns
     X : a matrix with the data points
     y : a vector with the class labels
    """

    X = np.loadtxt("{}/{}".format(dir_path, DATA_FILE), dtype = object)[1:,:]
    y = np.loadtxt("{}/{}".format(dir_path, PHENO_FILE), dtype = object)[1:,:]
    if sum(y[:,0]!=X[:,0])==0:
        return (X,y)
    else: 
        return(test_X.sort(axis=0), test_y.sort(axis=0))

def obtain_performance_metrics(y_true, y_pred): 
    """
    Function obtain_performance_metrics
    Receives two numpy arrays with the true and predicted labels.
    Computes all classification performance metrics.
    
    In this function you might call the functions:
    compute_accuracy(), compute_precision(), compute_recall()
    from the evaluation.py file. You can call them by writing:
    evaluation.compute_accuracy, and similarly.

    Returns a vector with one value per metric. The positions in the
    vector match the metric names in PERF_METRICS.
    """

    acc = evaluation.compute_accuracy(y_true, y_pred)
    prec = evaluation.compute_precision(y_true, y_pred)
    rec = evaluation.compute_recall(y_true, y_pred)
    perf = [acc, prec, rec]
    
    return perf



if __name__ == '__main__':

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
        "--testdir",
        required=True,
        help="Path to directory containing test data"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output will be created"
    )
    parser.add_argument(
        "--mink",
        required=True,
        help="Minimum value of k for k-NN algorithm"
    )
    parser.add_argument(
        "--maxk",
        required=True,
        help="Maximum value of k for k-NN algorithm"
    )
    
    args = parser.parse_args()
    
    #set arguments:
    trainpath = args.traindir
    testpath = args.testdir
    min_k = int(args.mink)
    max_k = int(args.maxk)

    # If the output directory does not exist, then create it
    out_dir = args.outdir
    os.makedirs(args.outdir, exist_ok=True)
  
    #Read the training and test data. For each dataset, get also the true labels.  
    train_X, train_y = load_data(trainpath)
    test_X, test_y = load_data(testpath)

    # Create the output file
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)
    
    # Write header for output file
    f_out.write('{}\t{}\t{}\t{}\n'.format('Value of k', 
                'Accuracy',
                'Precision',
                'Recall'))
   
  
    ############################## KNN algorithm ####################################

    # Create the k-NN object. 
    knn = KNNClassifier(train_X[:,1:], train_y[:,1:], metric = 'euclidean')

    # Iterate through all possible values of k:
    for k in range(min_k,max_k+1):
        
        knn.set_k(k)
        
        # 1. Perform KNN training and classify all the test points. In this step, you will
        # obtain a prediction for each test point. 
        
        y_pred = []
        
        for i in range(test_X.shape[0]):
            result = knn.predict(test_X[i,1:])
            if result:
                y_pred.append(result)
            else:
                knn.set_k(k-1)
                y_pred.append(knn.predict(test_X[i,1:]))
                knn.set_k(k)
                
    
        y_pred= np.array(y_pred)
        
        # 2. Compute performance metrics given the true-labels vector and the predicted-
        # labels vector (you might consider to use obtain_performance_metrics() function)
        
        perf = obtain_performance_metrics(test_y[:,1], y_pred)
    
        # 3. Write performance results in the output file, as indicated the in homework
        # sheet. 

        f_out.write('{}\t{}\t{}\t{}\n'.format(k, round(perf[0],2), round(perf[1],2), round(perf[2],2)))
                
    #Close the file       
    f_out.close()

