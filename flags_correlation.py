# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:13:04 2020

@author: larsh
"""

# Script for generating correlation matrix

# Libraries
from pandas import DataFrame
import numpy as np
from flags_load_data import X, attributeNames
import seaborn as sn
import matplotlib.pyplot as plt

# Change data matrix from object to float type
N = len(X)
M = len(X[0])
X_float = np.zeros((len(X), len(X[0])))

for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        X_float[i,j] = X[i,j]

# Create empty dictionary list
Data = {}

# Every attributes and corresponding values are inserted as a dictionary in the data list
for c in range(len(X[0])):
    
    variable = attributeNames[c]
    index = np.where(attributeNames==variable)[0][0]
    Data[variable] = X_float[:,index]


# Dataframe is created 
df = DataFrame(Data,columns=attributeNames)

# Correlation matrix is created
corrMatrix = df.corr()
print (corrMatrix)

# Generating the heatmap (correlation matrix)
sn.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(30,30))
sn.heatmap(corrMatrix, annot=False, linewidths=.5, ax=ax, vmin=-1,vmax=1, cmap="RdBu")


