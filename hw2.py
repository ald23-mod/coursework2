"""M3C 2018 Homework 2
Anas Lasri Doukkali CID: 01209387"""

import numpy as np
import matplotlib.pyplot as plt
from m1 import nmodel as nm #assumes that hw2_dev.f90 has been compiled with: f2py -c hw2_dev.f90 -m m1
from scipy.optimize import minimize

def read_data(tsize=60000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 70000 matrix, X
    and the corresponding labels are stored in a 70000 element array, y.
    The final 70000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int)%2 #rescale the image, convert the labels to 0s and 1s (For even and odd integers)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")

    return X,y,X_test,y_test
#----------------------------

def snm_test(X,y,X_test,y_test,omethod,input=(None)):
    """ In this part of the code I will discussed as asked the advantages of
    working with python and fortran instead of working just with python
    We need to understand the tradeoff that happens when using either fortran or
    python by itself or fortran by itself.
    When using Python you give up performance for productivity which means your
    code will be slower than a compiled language such as Fortran however as a tradeoff
    your productivity will be higher, this means it will take less lines of code than
    Fortran to accomplish the same task. However, this tradeoff when using Python
    varies. For example when using, fortran and making linear algebra we make use of
    LAPACK package which helps us speed up our calculations from the case of having
    done them in python. when we use python with Fortran with f2py we manage to
    get the best of both world
    So both of the key points that should be taken into account are productivity
    and performance .
    """
    """Train single neuron model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    X,y,X_test,y_test = read_data()
    n = X.shape[0]   #number of pixels, ie, 784
    d = X.shape[1]   #number of training pictures
    t = X_test.shape[1] #number of testing pictures
    nm.data_init(n,d) #intialize nm_x and nm_y for training model
    nm.nm_y = y
    nm.nm_x = X
    weights_init_guess = np.random.randn((n+1)) #initial guess for weights
    fvec = nm.sgd(weights_init_guess, n, 0, d, 0.1) #we use sgd to optimize our initial guess
    if omethod == 1:
        print("scipy minimize is the method being used")
        result=minimize(nm.snmodel, fvec,args=(d),method='L-BFGS-B',jac=True)
        fvec_optimized=result.x

    if omethod == 2:
        print("sgd is the method being used")
        fvec_optimized = nm.sgd(fvec, n, 0, d, 0.1)

    #We now proceed with testing our model and hence we
    #need to initialize our testing data set. For this we use the data_init function
    nm.data_init(n, d) #Initialize data now for test picturesto be able to perform the test
    nm.nm_y = y_test
    nm.nm_x = X_test
    alpha = nm.a_value(fvec_optimized, t)

    for i in range(t):
        alpha[i]=round(alpha[i])

    number_correct = 0

    for i in range(t):
        if alpha[i] == y_test[i]:
            number_correct = number_correct + 1

    test_error = 1-(number_correct/t)
    #output = (None) #output tuple, modify as needed

    return fvec_optimized,test_error
#--------------------------------------------

def nnm_test(X,y,X_test,y_test,m,omethod,input=(None)):
    """Train neural network model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of neurons in inner layer
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    fvec = np.random.randn(m*(n+2)+1) #initial fitting parameters

    #Add code to train NNM and evaluate testing error, test_error

    fvec_f = None #Modify to store final fitting parameters after training
    test_error = None #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def nm_analyze():
    """ Analyze performance of single neuron and neural network models
    on even/odd image classification problem
    Add input variables and modify return statement as needed.
    Should be called from
    name==main section below
    """

    return None
#--------------------------------------------

def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    output = nm_analyze()
