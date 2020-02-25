

import numpy as np

# Start by loading flags data
# "classification format":
from flags_load_data import Xstand, attributeNames
import matplotlib.pyplot as plt
from scipy.linalg import svd


# Dette gøres for at ændre matrixens værdiger fra ojekter til float
N = len(Xstand)
M = len(Xstand[0])
Xstand_float = np.zeros((len(Xstand), len(Xstand[0])))

for i in range(0, len(Xstand)):
    for j in range(0, len(Xstand[0])):
        Xstand_float[i,j] = Xstand[i,j]




# Subtract mean value from data
Y = Xstand_float - np.ones((N,1))*Xstand_float.mean(axis=0)

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