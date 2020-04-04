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
slettes = ['BOTR3','BOTR5']

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
CV = model_selection.KFold(K, shuffle=False)
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

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    #mu[k, :] = np.mean(X_train[:, 1:], 0)
    #sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    #X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    #X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train


    for i in range(len(lambdas)): 
        l = lambdas[i]
        # Estimate weights for the value of lambda, on entire training set
        lambdaI = l * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze() ##################################
        # Compute mean squared error with regularization with lambda
        Error_train_rlr[i,k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[i,k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
      
    k=k+1
    
# Bestemmer middelværdien for error i både train- og testsæt, 
# for hver lambda værdi, over alle de K folds
Error_test_rlr_mu = np.zeros(len(Error_test_rlr))
for n in range(len(Error_test_rlr)):
    Error_test_rlr_mu[n] = np.mean(Error_test_rlr[n,:])

print(Error_test_rlr_mu)

Error_train_rlr_mu = np.zeros(len(Error_train_rlr))
for n in range(len(Error_train_rlr)):
    Error_train_rlr_mu[n] = np.mean(Error_train_rlr[n,:])

print(Error_train_rlr_mu)

        
subplot(1,1,1)
loglog(lambdas,Error_train_rlr_mu.T,'b.-',lambdas,Error_test_rlr_mu.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
show()