# -*- coding: utf-8 -*-
"""
02450 Introduction to Machine Learning and Data Mining
Project 1

Created on Tue Feb 18 15:37:34 2020

@author: larsh
"""

import numpy as np
import pandas as pd
import math

# Load the flags data using the Pandas library
filename = 'flag.data'
df = pd.read_csv(filename, header=None)
bnp = pd.read_csv('bnp.txt',header=None)
bnp = bnp.get_values()

    
# Convert the dataframe to numpy arrays
raw_data = np.concatenate((df.get_values(),bnp[:,1]),axis=1)

# Making the data matrix X by indexing into data.
cols = range(0, 31) 
X = raw_data[:, cols]
X2 = X
# The attribute names are not stored in the data set, manually defined here
attributeNames = np.array(['NAME', 'LAMA', 'ZONE', 'AREA', 'POPU', 'LANG', 'RELI', 'BARS', 'STRI', 'COLO', 'RED', 'GREE', 'BLUE', 'YELL', 'WHIT', 'BLAC', 'ORAN', 'MAIN', 'CIRC', 'CROS', 'SALT', 'QUAR', 'SUNS', 'CRES', 'TRIA', 'ICON', 'ANIM', 'TEXT', 'TOPL', 'BOTR'])
attributeNames2 = attributeNames

# Colors is extracted from the last row which has all 8 colors and stored uniquely in a dictionary
colorLabel = raw_data[:,-1]
colorNames = np.unique(colorLabel)
colorDict = dict(zip(colorNames,range(len(colorNames))))

# The attributes with colors are now replaced by numbers according to dictionary
variable = ['MAIN', 'TOPL', 'BOTR']
for n in variable:
    X[:, np.where(attributeNames==n)[0][0]] = np.array([colorDict[cl] for cl in X[:, np.where(attributeNames==n)[0][0]]])



# One-out-of-K coding for the relevant attributes and make K vector
variable = ['LAMA','ZONE','LANG','RELI','MAIN', 'TOPL', 'BOTR']
K_v = np.zeros(len(variable))
u=0

#variable = ['TOPL']
for z in variable:
    
    cat = np.array(X[:, np.where(attributeNames==z)], dtype=int).T
    K = cat.max()-cat.min()+1
    
    # save K vector
    K_v[u] = K
    u=u+1
    
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

X2_country = X2[:,0]
X2 = X2[:,1:].astype(float)

attributeNames2 = attributeNames2[1:]

X_country = X[:,0]
X = X[:,1:]
attributeNames = attributeNames[1:]

# Sletter kollone 60 da det er den række som har en farve der ikke findes og der for er en række fuld af 0'er
X=np.delete(X,60,1)
attributeNames=np.delete(attributeNames,60)

# Her standadiseres atributterne 
Xstand=np.zeros((len(X), len(X[0])))
for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        Xstand[i,j] = (X[i,j]-X[:,j].mean(axis=0))/np.std(X[:,j])


variable = np.asarray(variable)


# Forbenious norm på one-out-of-K, hvor søjlerne deles med sqrt(K)
forb = np.ones(len(attributeNames))
for p in variable:
    K_where = np.where(variable==p)[0][0]
    K = int(K_v[K_where])
    
    for w in range(0,K):
        test = p + str(w)
        if test=='TOPL2': continue #Denne eksisterer ikke fordi række 60 er slettet
        Xstand[:, np.where(attributeNames==test)[0][0]] = Xstand[:, np.where(attributeNames==test)[0][0]]/math.sqrt(K)
        
        
        