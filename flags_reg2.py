# exercise 8.1.1

import matplotlib.pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from flags_load_data import Xstand, attributeNames

import time

time_start = time.clock()

# Import split as used in other assignment
#from flags_reg import CV

var = np.where(attributeNames=='BNP')[0][0]

# Ændrer navnet på de standardiserede data
X = Xstand
X = np.delete(X,var,1)
attributeNames = np.delete(attributeNames,var)
y = Xstand[:,var]

# Sletter følgende kolonner, da disse giver en singular matrix
slettes = ['TOPL0', 'TOPL1', 'TOPL3', 'TOPL4', 'TOPL5', 'TOPL6', 'TOPL7','BOTR0','BOTR1','BOTR2','BOTR3','BOTR4','BOTR5','BOTR6','BOTR7']
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
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_test_ANN = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


### ANN ###
## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape

# Parameters for neural network classifier
n_replicates = 3        # number of networks trained in each k-fold
max_iter = 5000

hidden_units = [1,2,3]

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

### ANN END ###

# Generate empty vectors for storing optimal lambda and number of hidden units
opt_lambda_fold = np.zeros((K))
opt_h_fold = np.zeros((K))

k=0
for train_index, test_index in CV.split(X,y):
    
    
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = K
    
    
    ### RLR VALIDATE ####
    #opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    cvf = internal_cross_validation
    CV2 = model_selection.KFold(cvf, shuffle=False)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index2, test_index2 in CV2.split(X_train,y_train):
        X_train2 = X_train[train_index2]
        y_train2 = y_train[train_index2]
        X_test2 = X_train[test_index2]
        y_test2 = y_train[test_index2]
        
        # Standardize the training and set set based on training set moments
        mu2 = np.mean(X_train2[:, 1:], 0)
        sigma2 = np.std(X_train2[:, 1:], 0)
        
        X_train2[:, 1:] = (X_train2[:, 1:] - mu2) / sigma2
        X_test2[:, 1:] = (X_test2[:, 1:] - mu2) / sigma2
        
        # precompute terms
        Xty2 = X_train2.T @ y_train2
        XtX2 = X_train2.T @ X_train2
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX2+lambdaI,Xty2).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train2-X_train2 @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test2-X_test2 @ w[:,f,l].T,2).mean(axis=0)
        
        
        # Den her skal laves ligesom de ovenover train og test error
        error_ANN = np.empty((cvf,len(hidden_units)+1)) # make a list for storing generalizaition error in each loop
        for h in range(1,len(hidden_units)+1):
            
            ###### ANN MODEL BEGIN #####
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            print('Training model of type:\n\n{}\n'.format(str(model())))
             
            
            print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K))  
            print('\nInner crossvalidation fold: {0}/{1}'.format(f+1,cvf))
            print('\nNumber of hidden units: {0}'.format(h))
        
            # Extract training and test set for current CV fold, convert to tensors
            X_train2 = torch.Tensor(X_train2)
            y_train2 = torch.Tensor(y_train2)
            X_test2 = torch.Tensor(X_test2)
            y_test2 = torch.Tensor(y_test2)
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test2)
            
            # Determine errors and errors
            se = (y_test_est.float().data.numpy().flatten()-y_test2.float().data.numpy().flatten())**2 # squared error
            mse = (sum(se)/len(y_test2)) #mean
            error_ANN[f,h] = mse # store error rate for current CV fold 
            

            ###### ANN MODEL END #####
        
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    ### RLR VALIDATE END ####

    # Find optimal lambda for each fold and save them in a vector
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    opt_lambda_fold[k] = opt_lambda


    # Find optimal number of hidden units for each fold and save them in a vector
    opt_h = hidden_units[np.argmin(np.mean(error_ANN,axis=0))]
    opt_h_fold[k] = opt_h


    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

    ###### ANN MODEL 2 BEGIN #####
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_h), #M features to optimal n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_h, 1), # optimal n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
#    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float().data.numpy().flatten()-y_test.float().data.numpy().flatten())**2 # squared error
    mse = (sum(se)/len(y_test)) #mean
    Error_test_ANN[k] = mse # store error rate for current CV fold  
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

    ###### ANN MODEL END #####

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()


    k+=1

show()




y_est = y_test_est.data.numpy(); 
y_true = y_test.data.numpy();
axis_range = [np.min([y_est.flatten(), y_true])-1,np.max([y_est.flatten(), y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('BNP of a country: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()



# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

#print('Weights in last fold:')
#for m in range(M):
#    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))


# MAKE THE SUMMARY TABLE
table = np.zeros([K,6]).astype(object)

for i in range(0,K):
    table[i,0] = i+1
    table[i,1] = opt_h_fold[i]
    table[i,2] = Error_test_ANN[i][0]
    table[i,3] = opt_lambda_fold[i]
    table[i,4] = Error_test_rlr[i][0]
    table[i,5] = Error_test[i][0]


dash = '-' * 80


print(dash)
print('{:<10s}{:>18s}{:>33s}{:>18s}'.format('Outer fold','ANN','Linear regression','Baseline'))
print('{:<10s}{:>11s}{:>16s}{:>15s}{:>11s}{:>16s}'.format('i','h_i*','E_i^test','lambda_i*','E_i^test','E_i^test'))
print(dash)
for i in range(len(table)):
      print('{:<10d}{:>10d}{:>17f}{:>14d}{:>12f}{:>16f}'.format(int(table[i][0]),int(table[i][1]),table[i][2],int(table[i][3]),table[i][4],table[i][5]))
print(dash)
print('{:<10s}{:>27f}{:>26f}{:>16f}'.format('MSE',np.sqrt(np.mean(Error_test_ANN)),np.sqrt(np.mean(Error_test_rlr)),np.sqrt(np.mean(Error_test))))
print(dash)
print('MSE with model predicting each observation is zero, {0}'.format(np.mean(y_train.data.numpy()**2)))
print(dash)



time_elapsed = (time.clock() - time_start)
print('The script execution time was: ',time_elapsed/60,' min')




