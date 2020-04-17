# exercise 8.2.5
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib 
# =============================================================================
# from scipy.io import loadmat
# =============================================================================
import torch
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net, draw_neural_net, mcnemar
from scipy import stats
import statistics as ss

import time

time_start = time.process_time()


from flags_load_data import X2, attributeNames2

attributeNames2 = np.delete(attributeNames2,5)
attributeNames = list()
for k in range(0,len(attributeNames2)):
    attributeNames.append(attributeNames2[k].astype(str))
# read XOR DATA from matlab datafile


y = X2[:,5].astype(int)
X = np.delete(X2,5,1)



#attributeNames = ['LAMA', 'POPU']
classNames = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']
N, M = X.shape
C = len(classNames)




# =============================================================================
# #Downsample: X = X[1:20,:] y = y[1:20,:]
# N, M = X.shape
# C = 2
# =============================================================================

# Normalize data
X = stats.zscore(X);

# Parameters for neural network classifier
n_hidden_units = [1,2,3]     # number of hidden units
n_replicates = 2       # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)

# K-fold crossvalidation
K = 10          # only five folds to speed up this example
Kin = K
CV = model_selection.KFold(K, shuffle=True)
CVin = model_selection.KFold(Kin, shuffle=True)

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,4, figsize=(20,5))

# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Generate empty vectors for storing optimal lambda and number of hidden units
opt_lambda_fold = np.zeros((K))
opt_h_fold = np.zeros((K))

errors = [] # make a list for storing generalizaition error in each loop
errors_base = [] # make a list for storing generalizaition error in each loop
errors_inner = [] # make a list for storing generalizaition error in each loop
errors_base_inner = [] # make a list for storing generalizaition error in each loop

lambda_interval = np.logspace(-8, 8, 100)
train_error_rate_log_inner = np.zeros(len(lambda_interval))
test_error_rate_log_inner = np.zeros(len(lambda_interval))
coefficient_norm_log_inner = np.zeros(len(lambda_interval))
train_error_rate = np.zeros((K))
test_error_rate = np.zeros((K))
coefficient_norm = np.zeros((K))

y_est_stat_base = []# np.zeros((np.ceil(y.size/K).astype(int),K))
y_est_stat_log = []# np.zeros((np.ceil(y.size/K).astype(int),K))
y_est_stat_ann =[]# np.zeros((np.ceil(y.size/K).astype(int),K))
y_tru_stat =[]# np.zeros((np.ceil(y.size/K).astype(int),K))

for k, (train_index, test_index) in enumerate(CV.split(X,y)): 

    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).type(torch.LongTensor)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]).type(torch.LongTensor)
    
       
   
    
    for kin, (train_index_inner, test_index_inner) in enumerate(CVin.split(X_train,y_train)): 
        X_train_inner = torch.Tensor(X[train_index_inner,:])
        y_train_inner = torch.Tensor(y[train_index_inner]).type(torch.LongTensor)
        X_test_inner = torch.Tensor(X[test_index_inner,:])
        y_test_inner = torch.Tensor(y[test_index_inner]).type(torch.LongTensor)
        
        
        # Fit regularized logistic regression model to training data to predict 
        # the type of wine

        for l in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[l] )
            
            mdl.fit(X_train_inner, y_train_inner)
        
            y_train_est_log_inner = mdl.predict(X_train_inner).T
            y_test_est_log_inner = mdl.predict(X_test_inner).T
            
            train_error_rate_log_inner[l] = np.sum(y_train_est_log_inner != y_train_inner.data.numpy()) / len(y_train_inner)
            test_error_rate_log_inner[l] = np.sum(y_test_est_log_inner != y_test_inner.data.numpy()) / len(y_test_inner)
        
            w_est_log_inner = mdl.coef_[0] 
            coefficient_norm_log_inner[l] = np.sqrt(np.sum(w_est_log_inner**2))
        
        min_error = np.min(test_error_rate_log_inner)
        opt_lambda_idx = np.argmin(test_error_rate_log_inner)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        for h in range(1,len(n_hidden_units)+1):
            
            ###### ANN MODEL BEGIN #####
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to H hiden units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, C), # H hidden units to 1 output neuron
                                torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output   
                                #torch.nn.Sigmoid() # final tranfer function
                                )
            loss_fn = torch.nn.CrossEntropyLoss() ### torch cross entropy loss
            
            print('Training model of type:\n\n{}\n'.format(str(model())))
             
            
            print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K))  
            print('\nInner crossvalidation fold: {0}/{1}'.format(kin+1,Kin))
            print('\nNumber of hidden units: {0}'.format(h))

            y_base = np.zeros([K]).astype(object)
    
    
            # Calculate baseline as mode of the training set
            y_base_inner = ss.mode(y_train_inner.data.numpy())
        
            # Train the net on training data
            net_inner, final_loss_inner, learning_curve_inner = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss_inner))
            
            # Determine estimated class labels for test set
            y_sigmoid_inner = net_inner(X_test_inner)
        
            
            values, y_test_est2_inner = torch.max(y_sigmoid_inner, axis=1)
            
            # Determine probability of each class using trained network
            softmax_logits_inner = net_inner(torch.tensor(X_test_inner, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est_inner = (torch.max(softmax_logits_inner, dim=1)[1]).to(torch.uint8)
            
            # Determine baseline test set based on training set size and y_base value
            y_base_est_inner = np.matlib.repmat(y_base_inner,y_test_inner.data.numpy().size,1)
        
        
            
            # Determine errors and errors
            y_test_inner = y_test_inner.to(torch.uint8)
        
            e_inner = y_test_est_inner != y_test_inner
            error_rate_inner = (sum(e_inner).type(torch.float)/len(y_test_inner)).data.numpy()
            errors_inner.append(error_rate_inner) # store error rate for current CV fold 
            
            e_base_inner =  np.equal(y_base_est_inner.flatten(),y_test_inner.data.numpy().flatten())
            error_rate_base_inner = sum(e_base_inner)/len(y_test_inner)
            errors_base_inner.append(error_rate_base_inner) # store error rate for current CV fold 
            
            
            ###### ANN MODEL END #####     

    # Find optimal number of hidden units for each fold and save them in a vector
    opt_h = n_hidden_units[np.argmin(np.mean(errors_base_inner,axis=0))]
    opt_h_fold[k] = opt_h
    
    
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda )
        
    mdl.fit(X_train, y_train)

    y_train_est_log = mdl.predict(X_train).T
    y_test_est_log = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est_log != y_train.data.numpy()) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est_log != y_test.data.numpy()) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    min_error = np.min(test_error_rate)
    
    
    ###### ANN MODEL BEGIN #####
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_h), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_h, C), # H hidden units to 1 output neuron
                        torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output   
                        #torch.nn.Sigmoid() # final tranfer function
                        )
    loss_fn = torch.nn.CrossEntropyLoss() ### torch cross entropy loss
        
    
    # Calculate baseline as mode of the training set
    y_base = ss.mode(y_train.data.numpy())
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)

    
    values, y_test_est2 = torch.max(y_sigmoid, axis=1)
    
    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).to(torch.uint8)
    
    # Determine baseline test set based on training set size and y_base value
    y_base_est = np.matlib.repmat(y_base,y_test.data.numpy().size,1)


    
    # Determine errors and errors
    y_test = y_test.to(torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) # store error rate for current CV fold 
    
    e_base =  np.equal(y_base_est.flatten(),y_test.data.numpy().flatten())
    error_rate_base = sum(e_base)/len(y_test)
    errors_base.append(error_rate_base) # store error rate for current CV fold 


    y_est_stat_base.append(y_base_est) 
    y_est_stat_log.append(y_test_est_log) 
    y_est_stat_ann.append(y_test.data.numpy()) 
    y_tru_stat.append(y_test.data.numpy())
    ###### ANN MODEL END #####
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set(xlim=(1/2, K+1/2), ylim=(0, 1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates ANN')

# Display the error rate across folds
summaries_axes[2].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors_base)), color=color_list)
summaries_axes[2].set_xlabel('Fold');
summaries_axes[2].set_xticks(np.arange(1, K+1))
summaries_axes[2].set(xlim=(1/2, K+1/2), ylim=(0, 1))
summaries_axes[2].set_ylabel('Error rate');
summaries_axes[2].set_title('Test misclassification rates baseline')

# Display the error rate across folds
summaries_axes[3].bar(np.arange(1, K+1), np.squeeze(np.asarray(test_error_rate)), color=color_list)
summaries_axes[3].set_xlabel('Fold');
summaries_axes[3].set_xticks(np.arange(1, K+1))
summaries_axes[3].set(xlim=(1/2, K+1/2), ylim=(0, 1))
summaries_axes[3].set_ylabel('Error rate');
summaries_axes[3].set_title('Test misclassification rates Logistic Regression')

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attributeNames,classNames)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

alpha = 0.05
[thetahat1, CI1, p1] = mcnemar(numpy.concatenate(y_tru_stat, axis=0 ), numpy.concatenate(y_est_stat_ann, axis=0 ), numpy.concatenate(y_est_stat_log, axis=0 ), alpha=alpha)

[thetahat2, CI2, p2] = mcnemar(numpy.concatenate(y_tru_stat, axis=0 ), numpy.concatenate(y_est_stat_ann, axis=0 ), numpy.concatenate(y_est_stat_base, axis=0 ).T, alpha=alpha)

[thetahat3, CI3, p3] = mcnemar(numpy.concatenate(y_tru_stat, axis=0 ), numpy.concatenate(y_est_stat_log, axis=0 ), numpy.concatenate(y_est_stat_base, axis=0 ).T, alpha=alpha)


# MAKE THE SUMMARY TABLE
table = np.zeros([K,6]).astype(object)

for i in range(0,K):
    table[i,0] = i+1
    table[i,1] = opt_h_fold[i]
    table[i,2] = errors[i]  # ANN
    table[i,3] = opt_lambda_fold[i]
    table[i,4] = test_error_rate[i] # Log
    table[i,5] = errors_base[i]  # Base


dash = '-' * 80


print(dash)
print('{:<10s}{:>18s}{:>33s}{:>18s}'.format('Outer fold','ANN','Linear regression','Baseline'))
print('{:<10s}{:>11s}{:>16s}{:>15s}{:>11s}{:>16s}'.format('i','h_i*','E_i^test','lambda_i*','E_i^test','E_i^test'))
print(dash)
for i in range(len(table)):
      print('{:<10d}{:>10d}{:>17f}{:>14d}{:>12f}{:>16f}'.format(int(table[i][0]),int(table[i][1]),table[i][2],int(table[i][3]),table[i][4],table[i][5]))
print(dash)



print('Ran Exercise Magnus')
