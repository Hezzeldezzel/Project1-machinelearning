# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:31:19 2020

@author: larsh
"""

from flags_pca import Z
from flags_load_data import X, attributeNames

from matplotlib.pyplot import figure, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt
import numpy as np


# Indices of the principal components to be plotted
i = 0
j = 1


## Begin plot
plt.title('Flags data: PCA')
plt.scatter(x=Z[:, i],
                y=Z[:, j], 
                s=50, alpha=0.5, color = 'b')
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()




# Colored by landmass
# INPUT
catNames = ['N. America','S. America','Europe','Africa','Asia','Oceania']
variable = 'LAMA'


# Defines the used colors in legens
color = ['r','g', 'b', 'c', 'm', 'y'] 

## Begin plot
plt.title('Flags data: PCA, colored by landmass')
for c in range(len(catNames)):
    idx = variable + str(c)
    
    class_mask = X[:, np.where(attributeNames==idx)[0][0]]==1

    plt.scatter(x=Z[class_mask, i],
                y=Z[class_mask, j], 
                s=50, alpha=0.5, color = color[c], label=catNames[c])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()


# Colored by language
# INPUT
catNames = ['English', 'Spanish', 'French', 'German', 'Slavic', 'Other Indo-European', 'Chinese', 'Arabic', 'Japanese/Turkish/Finnish/Magyar', 'Others']
variable = 'LANG'

## Begin plot
plt.title('Flags data: PCA, colored by language')
for c in range(len(catNames)):
    idx = variable + str(c)
    
    class_mask = X[:, np.where(attributeNames==idx)[0][0]]==1

    plt.scatter(x=Z[class_mask, i],
                y=Z[class_mask, j], 
                s=50, alpha=0.5, label=catNames[c])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()


# Colored by religion
# INPUT
catNames = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']
variable = 'RELI'

## Begin plot
plt.title('Flags data: PCA, colored by religion')
for c in range(len(catNames)):
    idx = variable + str(c)
    
    class_mask = X[:, np.where(attributeNames==idx)[0][0]]==1

    plt.scatter(x=Z[class_mask, i],
                y=Z[class_mask, j], 
                s=50, alpha=0.5, label=catNames[c])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()


# Colored by if the have the color red
# INPUT
catNames = ['Not red in flag', 'Red in flag']
variable = 'RED'

color = ['lightgray','r']

## Begin plot
plt.title('Flags data: PCA, colored by red in flag')
for c in range(len(catNames)):
    idx = variable
    
    class_mask = X[:, np.where(attributeNames==idx)[0][0]]==c

    plt.scatter(x=Z[class_mask, i],
                y=Z[class_mask, j], 
                s=50, alpha=0.5, label=catNames[c], color = color[c])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()