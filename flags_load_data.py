# -*- coding: utf-8 -*-
"""
02450 Introduction to Machine Learning and Data Mining
Project 1

Created on Tue Feb 18 15:37:34 2020

@author: larsh
"""

import numpy as np
import pandas as pd


# Load the flags data using the Pandas library
filename = 'flag.data'
df = pd.read_csv(filename, header=None)

# Convert the dataframe to numpy arrays
raw_data = df.get_values()

# Making the data matrix X by indexing into data.
cols = range(0, 30) 
X = raw_data[:, cols]

# The attribute names are not stored in the data set, manually defined here
attributeNames = np.array(['NAME', 'LAMA', 'ZONE', 'AREA', 'POPU', 'LANG', 'RELI', 'BARS', 'STRI', 'COLO', 'RED', 'GREE', 'BLUE', 'YELLO', 'WHIT', 'BLAC', 'ORAN', 'MAIN', 'CIRC', 'CROS', 'SALT', 'QUAR', 'SUNS', 'CRES', 'TRIA', 'ICON', 'ANIM', 'TEXT', 'TOPL', 'BOTR'])


# Colors is extracted from the last row which has all 8 colors and stored uniquely in a dictionary
colorLabel = raw_data[:,-1]
colorNames = np.unique(colorLabel)
colorDict = dict(zip(colorNames,range(len(colorNames))))

# The attributes with colors are now replaced by numbers according to dictionary
variable = ['MAIN', 'TOPL', 'BOTR']
for n in variable:
    X[:, np.where(attributeNames==n)[0][0]] = np.array([colorDict[cl] for cl in X[:, np.where(attributeNames==n)[0][0]]])


# One-out-of-K coding for the relevant attributes
variable = ['LAMA','ZONE','LANG','RELI','MAIN', 'TOPL', 'BOTR']
#variable = ['TOPL']
for z in variable:
    cat = np.array(X[:, np.where(attributeNames==z)], dtype=int).T
    K = cat.max()-cat.min()+1
    cat = cat-cat.min()
    cat_encoding = np.zeros((cat.size, K))
    cat_encoding[np.arange(cat.size), cat] = 1
    
    # The new one-out-of-K coding is inserted at the origianal column in the X-matrix. The X-matrix has now more columns than before
    X = np.concatenate( (X[:, :np.where(attributeNames==z)[0][0]], cat_encoding, X[:, np.where(attributeNames==z)[0][0]+1:]), axis=1)
        
    # The attribute names are now updated to as many K's
    insert_attribute = np.empty(K,dtype='<U6')
    for x in range(0,K):
        insert_attribute[x] = z + str(x)
        
    attributeNames = np.concatenate( (attributeNames[:np.where(attributeNames==z)[0][0]],insert_attribute,attributeNames[np.where(attributeNames==z)[0][0]+1:]),axis=0)


# A combined matrix with header
X_c = np.insert(X, 0 ,attributeNames, 0)

X_country = X[1,:]
X = X[:,1:]

#for i in range(0,len(X[0])):
#    print(attributeNames[i])
#    print(str(X[0,i]))