

import numpy as np

# Start by loading flags data
# "classification format":
from flags_load_data import X, attributeNames
import matplotlib.pyplot as plt
from scipy.linalg import svd



N = len(X)
M = len(X[0])
X = X[:,1:]
G = np.zeros((len(X), len(X[0])))

for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        G[i,j] = X[i,j]




# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('Ran Exercise 2.1.3')