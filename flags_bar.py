# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:32:28 2020

@author: larsh
"""
# Import library and dataset
from flags_load_data import X, attributeNames, colorNames, colorDict
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from scipy import stats
import math



# Number of samples
N = len(X[:,1])



############################################################################################

category = ['RED', 'GREE', 'BLUE', 'YELL', 'WHIT', 'BLAC', 'ORAN']
color = ['r','g','b','y','w','k','tab:orange']

cont = np.zeros(len(category))

for c in range(0,len(category)):
    variable = category[c]
    index = np.where(attributeNames==variable)[0][0]
    sum_int = np.sum(X[:,index])
    cont[c] = sum_int/N*100
    

category = ['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black', 'Orange']
plt.bar(category, cont, align='center', alpha=0.5, color=color, edgecolor='k')
plt.xticks(category)
plt.ylabel('Percentage of flags with color present')
plt.show()

#############################################################################################


#category = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']

category = ['RELI0', 'RELI1', 'RELI2', 'RELI3', 'RELI4', 'RELI5', 'RELI6', 'RELI7']

find1 = 'CROS'
find2 = 'SALT'

index_find1 = np.where(attributeNames==find1)[0][0]
index_find2 = np.where(attributeNames==find2)[0][0]


cont = np.zeros(len(category))

for c in range(0,len(category)):
    for i in range(0,N):
        variable = category[c]
        index = np.where(attributeNames==variable)[0][0]
        
        # Find all the rows where RELIX is present
        if X[i,index] == 1:
            # Check if they have a CROS
            if X[i,index_find1] == 1:
                cont[c] = cont[c]+1
            # Check if they have SALT
            if X[i,index_find2] == 1:
                cont[c] = cont[c]+1
    cont[c] = cont[c]/N*100

category = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']

plt.bar(category, cont, align='center', alpha=0.5, edgecolor='k', color='b')
plt.xticks(category, rotation =45)
plt.ylabel('Percentage')
plt.show()



#############################################################################################


#category = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']

category = ['RELI0', 'RELI1', 'RELI2', 'RELI3', 'RELI4', 'RELI5', 'RELI6', 'RELI7']

find1 = 'SUNS'

index_find1 = np.where(attributeNames==find1)[0][0]


cont = np.zeros(len(category))

for c in range(0,len(category)):
    for i in range(0,N):
        variable = category[c]
        index = np.where(attributeNames==variable)[0][0]
        
        # Find all the rows where RELIX is present
        if X[i,index] == 1:
            # Check if they have a CROS
            if X[i,index_find1] == 1:
                cont[c] = cont[c]+1
    cont[c] = cont[c]/N*100

category = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']

plt.bar(category, cont, align='center', alpha=0.5, edgecolor='k', color='r')
plt.xticks(category, rotation =45)
plt.ylabel('Percentage')
plt.show()


