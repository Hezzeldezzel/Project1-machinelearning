# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:11:33 2020

@author: larsh
"""

# Import library and dataset
from flags_load_data import X, attributeNames, colorNames
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from scipy import stats
import math


variable = 'COLO'

index = np.where(attributeNames==variable)[0][0]

# Number of samples
N = len(X[:,index])

# Mean
mu = X[:,index].mean()

# Standard deviation
s = X[:,index].std(ddof=1)

# Number of bins in histogram
nbins = len(colorNames)

# Plot the histogram
f = figure()
title('Number of color in flags: Histogram and theoretical distribution')
hist(X[:,index], bins=nbins, density=True)
plt.ylabel('Distribution')
plt.xlabel('Number of colors in flags')

# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,index].min(), X[:,index].max(), 1000)
pdf = stats.norm.pdf(x,loc=3.2,scale=1)
plot(x,pdf,'.',color='red')

# Compute empirical mean and standard deviation
mu_ = X[:,index].mean()
s_ = X[:,index].std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

show()


#########################################################################

variable = 'STRI'

index = np.where(attributeNames==variable)[0][0]

# Number of samples
N = len(X[:,index])

# Mean
mu = X[:,index].mean()

# Standard deviation
s = X[:,index].std(ddof=1)

# Number of bins in histogram
nbins = math.ceil(1 + 3.222*math.log(len(X),10))

# Plot the histogram
f = figure()
hist(X[:,index], bins=nbins, density=True)
plt.ylabel('Distribution')
plt.xlabel('Number of stripes in flags')



    
    
    
    
    
    
    
    

