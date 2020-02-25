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

# A combined matrix with header
X_c = np.insert(X, 0 ,attributeNames, 0)


# One-out-of-K coding 
ZONE_cat = np.array(X[:, np.where(attributeNames=='ZONE')], dtype=int).T
ZONE_K = ZONE_cat.max()
ZONE_encoding = np.zeros((ZONE_cat.size, ZONE_K))
ZONE_encoding[np.arange(ZONE_cat.size), ZONE_cat-1] = 1

X2 = np.concatenate( (X[:, :2], ZONE_encoding, X[:, 3:]), axis=1)