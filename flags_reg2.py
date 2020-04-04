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

# Import split as used in other assignment
from flags_reg import CV

var = np.where(attributeNames=='AREA')[0][0]

# Ændrer navnet på de standardiserede data
X = Xstand
X = np.delete(X,var,1)
attributeNames = np.delete(attributeNames,var)
y = Xstand[:,var]

# Sletter følgende kolonner, da disse giver en singular matrix
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
#CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
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
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 2000

hidden_units = [1,2,3,4,5]

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

### ANN END ###


k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10  
    
    
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
        errors = [] # make a list for storing generalizaition error in each loop
        for h in range(1,len(hidden_units)+1):

            ###### INSERT ANN MODEL HERE #####
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
            errors.append(mse) # store error rate for current CV fold 
            
            # Display the learning curve for the best net in the current fold
            h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
            h.set_label('CV fold {0}'.format(k+1))
            summaries_axes[0].set_xlabel('Iterations')
            summaries_axes[0].set_xlim((0, max_iter))
            summaries_axes[0].set_ylabel('Loss')
            summaries_axes[0].set_title('Learning curves')
    
            
            
            ###### ANN MODEL END #####
        
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    ### RLR VALIDATE END ####

    # Display the MSE across folds
#    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
#    summaries_axes[1].set_xlabel('Fold')
#    summaries_axes[1].set_xticks(np.arange(1, K+1))
#    summaries_axes[1].set_ylabel('MSE')
#    summaries_axes[1].set_title('Test mean-squared-error')



    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    #mu[k, :] = np.mean(X_train[:, 1:], 0)
    #sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    #X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    #X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
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
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

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
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
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

print('Ran script flags_reg2.py')