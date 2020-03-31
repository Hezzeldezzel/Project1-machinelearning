# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

from flags_load_data import Xstand, attributeNames

var = np.where(attributeNames=='AREA')[0][0]

# Ændrer navnet på de standardiserede data
X = Xstand
X = np.delete(X,var,1)
attributeNames = np.delete(attributeNames,var)
y = Xstand[:,var]

# Sletter følgende kolonner, da denne giver en singular matrix
slettes = ['BOTR0','BOTR1','BOTR2','BOTR3','BOTR4','BOTR5','BOTR6','BOTR7']
#slettes = ['BOTR0','BOTR1','BOTR2','BOTR3','BOTR4','BOTR5','BOTR6','BOTR7','TOPL0', 'TOPL1',
#       'TOPL3', 'TOPL4', 'TOPL5', 'TOPL6', 'TOPL7','MAIN0', 'MAIN1', 'MAIN2', 'MAIN3',
#       'MAIN4', 'MAIN5', 'MAIN6', 'MAIN7','LAMA0', 'LAMA1', 'LAMA2', 'LAMA3', 'LAMA4', 'LAMA5',
 #      'ZONE0', 'ZONE1', 'ZONE2', 'ZONE3','TEXT','LANG0', 'LANG1', 'LANG2', 'LANG3', 'LANG4',
#       'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9', 'RELI0', 'RELI1',
#       'RELI2', 'RELI3', 'RELI4', 'RELI5', 'RELI6', 'RELI7','BARS','ORAN','BLAC','ICON']
for z in range(len(slettes)):
    o = slettes[z]
    var = np.where(attributeNames==o)[0][0]
    X = np.delete(X,var,1)
    attributeNames = np.delete(attributeNames,var)


N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.concatenate((['Offset'],attributeNames))
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((len(lambdas),K))
Error_test = np.empty((len(lambdas),K))
Error_train_rlr = np.empty((len(lambdas),K))
Error_test_rlr = np.empty((len(lambdas),K))
Error_train_nofeatures = np.empty((len(lambdas),K))
Error_test_nofeatures = np.empty((len(lambdas),K))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]


    for i in range(len(lambdas)): 
        l = lambdas[i]
        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = l * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze() ##################################
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[i,k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[i,k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
      
print(np.sum(sigma<0.001))

# Mangler at finde middelværdi af Error_train og Error_split for hver kolonne.
# Dette angiver error ved hver lambda værdi  valgte interval