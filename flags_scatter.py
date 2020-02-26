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


i = np.where(attributeNames=='AREA')[0][0]
j = np.where(attributeNames=='POPU')[0][0]


color = ['r','g', 'b', 'c', 'm', 'y']


plt.title('Area and population, colored by landmass')
for c in range(len(LAMANames)):
    plt.scatter(x=X[:, i],
                y=X[:, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=LAMANames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()






