# exercise 8.2.5
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib 
# =============================================================================
# from scipy.io import loadmat
# =============================================================================
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import statistics as ss

# =============================================================================
# # Load Matlab data file and extract variables of interest
# mat_data = loadmat('../Data/wine2.mat')
# attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
# X = mat_data['X']
# y = mat_data['y']
# 
# =============================================================================
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
n_hidden_units = 3     # number of hidden units
n_replicates = 5       # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)

# K-fold crossvalidation
K = 5            # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)
# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,3, figsize=(15,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model, see also Exercise 8.2.2-script for more information.
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, C), # H hidden units to 1 output neuron
                    torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output   
                    #torch.nn.Sigmoid() # final tranfer function
                    )
loss_fn = torch.nn.CrossEntropyLoss() ### torch cross entropy loss

y_base = np.zeros([K]).astype(object)

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
errors_base = [] # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).type(torch.LongTensor)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]).type(torch.LongTensor)
    
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
# =============================================================================
#     y_test_est = (y_sigmoid>.5).to(torch.uint8)#.flatten()
# =============================================================================
    
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

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attributeNames,classNames)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

print('Ran Exercise Magnus')