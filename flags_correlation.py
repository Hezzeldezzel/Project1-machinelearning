# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:13:04 2020

@author: larsh
"""

from pandas import DataFrame
import numpy as np
from flags_load_data import X, attributeNames
import seaborn as sn
import matplotlib.pyplot as plt

# Dette gøres for at ændre matrixens værdiger fra ojekter til float
N = len(X)
M = len(X[0])
X_float = np.zeros((len(X), len(X[0])))

for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        X_float[i,j] = X[i,j]

# Create zero vector
Data = {}

for c in range(len(X[0])):
    
    variable = attributeNames[c]
    index = np.where(attributeNames==variable)[0][0]
    Data[variable] = X_float[:,index]



df = DataFrame(Data,columns=attributeNames)


corrMatrix = df.corr()
print (corrMatrix)

sn.set(font_scale=2)
fig, ax = plt.subplots(figsize=(30,30))
sn.heatmap(corrMatrix, annot=False, linewidths=.5, ax=ax)



# Old:
#Data = {'A': X_float[:,np.where(attributeNames=='AREA')[0][0]],
#        'B': X_float[:,np.where(attributeNames=='POPU')[0][0]],
#        'C': X_float[:,np.where(attributeNames=='SUNS')[0][0]]
#        }

#df = DataFrame(Data,columns=['A','B','C'])