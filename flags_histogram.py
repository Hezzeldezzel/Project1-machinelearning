# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:11:33 2020

@author: larsh
"""

# Import library and dataset
from flags_load_data import X, attributeNames
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


variable = 'COLO'

df = X[:, np.where(attributeNames==variable)[0][0]]
 
# Make default histogram    
sns.distplot( df , kde=False)

plt.title('Number of colors in a flag')
plt.xlabel('# colors in flags')
plt.ylabel('Values')




#sns.plt.show()
 
# Control the number of bins
#sns.distplot( df, bins=20 )
#sns.plt.show()
