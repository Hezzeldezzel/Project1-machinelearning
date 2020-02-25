# -*- coding: utf-8 -*-
"""
02450 Introduction to Machine Learning and Data Mining
Project 1

Created on Tue Feb 18 15:37:34 2020

@author: larsh
"""

import numpy as np

# Start by loading flags data
# "classification format":
from flags_load_data import *

# Compute vectors with mean, standard deviation, median and range for each attribute of type ratio

index = np.where(attributeNames=='AREA')
stats_AREA = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='POPU')
stats_POPU = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='BARS')
stats_BARS = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='STRI')
stats_STRI = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='COLO')
stats_COLO = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='CIRC')
stats_CIRC = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='CROS')
stats_CROS = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='SALT')
stats_SALT = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='QUAR')
stats_QUAR = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]

index = np.where(attributeNames=='SUNS')
stats_SUNS = [X[:,index].mean(), X[:,index].std(ddof=1), np.median(X[:,index]), X[:,index].max()-X[:,index].min()]