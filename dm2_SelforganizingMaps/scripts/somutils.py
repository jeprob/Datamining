"""
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
# Author: Dean Bodenham, May 2016
# Modified by: Damian Roqueiro, May 2017
# Modified by: Christian Bock, April 2021
# Completed by: Jennifer Probst, Mai 2021

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from matplotlib.lines import Line2D

np.random.seed(3)

"""
A function to create the S curve
"""
def makeSCurve():
    n_points = 1000
    noise = 0.2
    X, color = datasets.make_s_curve(n_points, noise=noise, random_state=0)
    Y = np.array([X[:,0], X[:,2]])
    Y = Y.T
    # Stretch in all directions
    Y = Y * 2
    
    # Now add some background noise
    xMin = np.min(Y[:,0])
    xMax = np.max(Y[:,0])
    yMin = np.min(Y[:,1])
    yMax = np.max(Y[:,1])
    
    n_bg = n_points//10
    Ybg = np.zeros(shape=(n_bg,2))
    Ybg[:,0] = np.random.uniform(low=xMin, high=xMax, size=n_bg)
    Ybg[:,1] = np.random.uniform(low=yMin, high=yMax, size=n_bg)
    
    Y = np.concatenate((Y, Ybg))
    return Y


"""
Plot the data and SOM for the S-curve
  data: 2 dimensional dataset (first two dimensions are plotted)
  buttons: N x 2 array of N buttons in 2D
  fileName: full path to the output file (figure) saved as .pdf or .png
"""
def plotDataAndSOM(data, buttons, fileName):
    fig = plt.figure(figsize=(8, 8))
    # Plot the data in grey
    plt.scatter(data[:,0], data[:,1], c='grey')
    # Plot the buttons in large red dots
    plt.plot(buttons[:,0], buttons[:,1], 'ro', markersize=10)
    # Label axes and figure
    plt.title('S curve dataset, with buttons in red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(fileName)
   
    
"""
Plot the data and SOM for the S-curve
  data: 2 dimensional dataset (first two dimensions are plotted)
  buttons: N x 2 array of N buttons in 2D
  fileName: full path to the output file (figure) saved as .pdf or .png
"""
def plotReconstructionError(error, fileName):
    fig = plt.figure(figsize=(8, 8))
    # Plot reconstruction error
    t = range(error.size)    
    plt.scatter(t, error)
    # Label axes and figure
    plt.title('Reconstruction error for the SOM of the S curve dataset')
    plt.xlabel('iteration number t')
    plt.ylabel('Reconstruction error')
    plt.savefig(fileName)


"""
Create a grid of points, dim p x q, and save grid in a (p*q, 2) array
  first column: x-coordinate
  second column: y-coordinate
"""
def createGrid(p, q):
    index = 0
    grid = np.zeros(shape=(p*q, 2))
    for i in range(p):
        for j in range(q):
            index = i*q + j
            grid[index, 0] = i
            grid[index, 1] = j
    return grid


"""
A function to plot the crabs results
It applies a SOM previously computed (parameters grid and buttons) to a given
dataset (parameters data)

Parameters
 X : is the original data that was used to compute the SOM.
     Rows are samples and columns are features.
 idInfo : contains the information (sp and sex for the crab dataset) about
          each data point in X.
          The rows in idInfo match one-to-one to rows in X.
 grid, buttons : obtained from computing the SOM on X.
 fileName : full path to the output file (figure) saved as .pdf or .png
"""
def plotSOMCrabs(X, idInfo, grid, buttons, fileName):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(grid[:, 0], grid[:, 1], s=700, facecolors='none', edgecolors='k')
    #make jitter by adding gaussian noise to both axes
    xjitter = np.random.randn(X.shape[0]) * 0.1
    yjitter = np.random.randn(X.shape[0]) * 0.1
    for i in range(X.shape[0]):
        #find the coordinates of the nearest button
        button_ind = findNearestButtonIndex(X[i, :], buttons)
        b_x_coord = grid[button_ind, 0]
        b_y_coord = grid[button_ind, 1]
        #find the color of point
        if (idInfo[i, 0] == 'B' and idInfo[i, 1] == 'M'): c = '#0038ff'  #dark blue for blue male
        if (idInfo[i, 0] == 'B' and idInfo[i, 1] == 'F'): c = '#00eefd'  #cyan for blue female
        if (idInfo[i, 0] == 'O' and idInfo[i, 1] == 'M'): c = '#ffa22f'  #orange for orange male
        if (idInfo[i, 0] == 'O' and idInfo[i, 1] == 'F'): c = '#e9e824'  #yellow for orange female
        plt.scatter(b_x_coord + xjitter[i], b_y_coord + yjitter[i], c=c, label=c, s=2)
    plt.title('SOM applied to the crabs dataset')
    #make the legend
    custom_lines = [Line2D([0], [0], color='#0038ff', lw=4), Line2D([0], [0], color="#00eefd", lw=4),
                    Line2D([0], [0], color="#ffa22f", lw=4), Line2D([0], [0], color="#e9e824", lw=4)]
    #Get legend out of the plot
    b = ax.get_position()
    ax.set_position([b.x0, b.y0, b.width * 0.7, b.height])
    plt.legend(custom_lines, ['Blue Male', 'Blue Female', 'Orange Male', 'Orange Female'], bbox_to_anchor=(1, 1),
               loc='upper left', fontsize='x-small')
    plt.savefig(fileName)


"""
Input: Two arrays z0 and z1
Output: Their euclidean distance in grid space
"""
def getGridDist(z0, z1):
    return distance.euclidean(z0, z1)


"""
Input: Two arrays z0 and z1
Output: Their euclidean distance in feature space
"""
def getFeatureDist(z0, z1):
    return distance.euclidean(z0, z1)


"""
Input: 2 dimensional grid array with coordinates for k (=p*q) datapoints
Output: Distance matrix with distances between the datapoints in the grid
"""
def createGridDistMatrix(grid):
    k = grid.shape[0]
    gridmat = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            gridmat[i,j]= getGridDist(grid[i,:],grid[j,:])
    return gridmat


"""
Input: Max epsilon value, N: length of array
Output: Array for epsilons, the values in the array decrease from epsilon_max
     to 1
"""
def createEpsilonArray(epsilon_max, N):
    return np.arange(epsilon_max, 1, -(epsilon_max-1)/N) 


"""
Input: Max alpha value, lambda_ is used to modify how much the exponential 
    decay of alphas is, N: length of array
Output: Array for alphas, the values in the array decrease from alpha_max
    according to the equation in the homework sheet
"""
def createAlphaArray(alpha_max, lambda_, N):
    ts = np.arange(0, N)
    alphas = alpha_max*np.exp(-1*(lambda_*ts))
    return alphas


"""
Input: X is whole data set, K is number of buttons to choose, d the number of datapoints
Output: K randomly chosen datapoints from X
"""
def initButtons(X, K, d):
    indices = np.random.choice(d, K)
    return X[indices,:]


"""
Input: x is one data point, buttons is the grid in feature space
Output: index of the closest button to x in feature space
"""
def findNearestButtonIndex(x, buttons):
    d = [getFeatureDist(x, button) for button in buttons]
    return (d.index(min(d)))


"""
Input: index of the button to find the epsilon-neighbourhood from; distances of all buttons 
Output: boolean vector of length K: True if button in epsilon neighborhood
    (distance to index_button smaller than epsilon), False if not
"""
def findButtonsInNhd(index, epsilon, buttonDist):
    return [button<=epsilon for button in buttonDist[index,:]]


"""
Input: 
Output: 
Do gradient descent step, update each button position IN FEATURE SPACE
"""
def updateButtonPosition(button, x, alpha):
    return alpha*(x-button)
    

"""
Input: data and buttons
Output: Squared distance between data points and their nearest button
"""
def computeError(data, buttons):
    error=0
    for i in range(data.shape[0]):
        minimal_but= findNearestButtonIndex(data[i,:], buttons)
        error += getFeatureDist(data[i,:], buttons[minimal_but,:])**2 
    return error


"""
Implementation of the self-organizing map (SOM)

Parameters
 X : data, rows are samples and columns are features
 p, q : dimensions of the grid
 N : number of iterations
 alpha_max : upper limit for learning rate
 epsilon_max : upper limit for radius
 compute_error : boolean flag to determine if the error is computed.
                 The computation of the error is time-consuming and may
                 not be necessary every time the function is called.
 lambda_ : decay constant for learning rate
                 
Returns
 buttons, grid : the buttons and grid of the newly created SOM
 error : a vector with error values. This vector will contain zeros if 
         compute_error is False

"""
def SOM(X, p, q, N, alpha_max, epsilon_max, compute_error=False, lambda_=0.01):
    
    # 1. Create grid and compute pairwise distances
    grid = createGrid(p, q) #2dim array with coordinates
    gridDistMatrix = createGridDistMatrix(grid) #matrix of dim (p*q)*(p*q)
    
    # 2. Randomly select K out of d data points as initial positions of the buttons
    K = p * q #total number of datapoints
    d = X.shape[0] #dimension of the data
    buttons = initButtons(X, K, d)
    
    # 3. Create a vector of size N for learning rate alpha
    learning_rates = createAlphaArray(alpha_max, lambda_, N)
    
    # 4. Create a vector of size N for epsilon 
    epsilons = createEpsilonArray(epsilon_max, N)

    # Initialize a vector with N zeros for the error
    # This vector may be returned empty if compute_error is False
    error = np.zeros(N)

    # 5. Iterate N times
    for i in range(N):
        # 6. Initialize/update alpha and epsilon
        epsilon = epsilons[i]
        alpha = learning_rates[i]
        
        # 7. Choose a random index t in {1, 2, ..., d}
        index_t = int(np.random.choice(range(1,d), 1))
        
        # 8. Find button m_star that is nearest to x_t in feature space
        button_mstar_ind = findNearestButtonIndex(X[index_t,:], buttons)

        # 9. Find all indices of grid points in epsilon-nhd of m_star in grid space
        neighbors_mstar = findButtonsInNhd(button_mstar_ind, epsilon, gridDistMatrix)

        # 10. Update position (in FEATURE SPACE) of all buttons m_j
        #     in epsilon-nhd of m_star, including m_star
        for j in range(K):
            if neighbors_mstar[j]: #update if in neighborhood
                buttons[j] += updateButtonPosition(buttons[j], X[index_t,:], alpha)
                
        # Compute the error 
        if compute_error:
            error[i] = computeError(X, buttons)

    # 11. Return buttons, grid and error
    return (buttons, grid, error)    