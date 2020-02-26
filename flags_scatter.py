# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:52:27 2020

@author: larsh
"""




## SCATTERPLOT

from flags_load_data import X, attributeNames

import numpy as np
import matplotlib.pyplot as plt

## Scatterplot colored by landmass


# Landmass names
LAMANames = ['N. America','S. America','Europe','Africa','Asia','Oceania']


attributeNames_c = attributeNames.copy();


# Finds the columns of area and population
i = np.where(attributeNames=='AREA')[0][0]
j = np.where(attributeNames=='POPU')[0][0]


# Defines the used colors in legens
color = ['r','g', 'b', 'c', 'm', 'y'] 

## Begin plot
plt.title('Area and population, colored by landmass')
for c in range(len(LAMANames)):
    
    # Makes a variable for every c which is used to find the specific columns
    idx = 'LAMA' + str(c)
    
    # The column for each landmass inclued 0's and 1's, which are multiplied for
    # each loop to the columsn for area and population, respectively
    plt.scatter(x=X[:, np.where(attributeNames==idx)[0][0]]*X[:,i],
                y=X[:, np.where(attributeNames==idx)[0][0]]*X[:,j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=LAMANames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()






