# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:04:54 2020

@author: larsh
"""

# library & dataset
from flags_load_data import X, attributeNames
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

variable = 'SUNS'

df = X[:, np.where(attributeNames==variable)[0][0]]
 
# Make boxplot for one group only
sns.boxplot( y=df )

plt.title('Number of sun or star symbols in flag')
plt.ylabel('# of suns or star symbols in flag')

           

#sns.plt.show()
