# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:05:19 2020

@author: Mads Dalum Hesseldah
"""

import matplotlib.pyplot as plt
import numpy as np
from flags_load_data import attributeNames, Xstand
from flags_pca import V

from scipy.linalg import svd

N,M = Xstand.shape

V=V.T


# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = 0.2
r = np.arange(1,M+1)

plt.rcParams['figure.figsize'] = [10, 6] # for square canvas
#plt.rcParams['figure.subplot.left'] = 0
#plt.rcParams['figure.subplot.bottom'] = 0
#plt.rcParams['figure.subplot.right'] = 1
#plt.rcParams['figure.subplot.top'] = 1

for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

lm = np.arange(np.where(attributeNames=='LAMA0')[0][0]+1,np.where(attributeNames=='LAMA5')[0][0]+2)

for i in pcs:    
    plt.bar(lm+i*bw, V[np.where(attributeNames=='LAMA0')[0][0]:np.where(attributeNames=='LAMA5')[0][0]+1,i], width=bw)
plt.xticks(lm+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Landmass: PCA Component Coefficients')
plt.show()


lg = np.arange(np.where(attributeNames=='LANG0')[0][0]+1,np.where(attributeNames=='LANG9')[0][0]+2)

for i in pcs:    
    plt.bar(lg+i*bw, V[np.where(attributeNames=='LANG0')[0][0]:np.where(attributeNames=='LANG9')[0][0]+1,i], width=bw)
plt.xticks(lg+bw, attributeNames[np.where(attributeNames=='LANG0')[0][0]:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Language: PCA Component Coefficients')
plt.show()

re = np.arange(np.where(attributeNames=='RELI0')[0][0]+1,np.where(attributeNames=='RELI7')[0][0]+2)

for i in pcs:    
    plt.bar(re+i*bw, V[np.where(attributeNames=='RELI0')[0][0]:np.where(attributeNames=='RELI7')[0][0]+1,i], width=bw)
plt.xticks(re+bw, attributeNames[np.where(attributeNames=='RELI0')[0][0]:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Religion: PCA Component Coefficients')
plt.show()

ma = np.arange(np.where(attributeNames=='MAIN0')[0][0]+1,np.where(attributeNames=='MAIN7')[0][0]+2)

for i in pcs:    
    plt.bar(ma+i*bw, V[np.where(attributeNames=='MAIN0')[0][0]:np.where(attributeNames=='MAIN7')[0][0]+1,i], width=bw)
plt.xticks(ma+bw, attributeNames[np.where(attributeNames=='MAIN0')[0][0]:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Mainhue: PCA Component Coefficients')
plt.show()

to = np.arange(np.where(attributeNames=='TOPL0')[0][0]+1,np.where(attributeNames=='TOPL7')[0][0]+2)

for i in pcs:    
    plt.bar(to+i*bw, V[np.where(attributeNames=='TOPL0')[0][0]:np.where(attributeNames=='TOPL7')[0][0]+1,i], width=bw)
plt.xticks(to+bw, attributeNames[np.where(attributeNames=='TOPL0')[0][0]:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Topleft: PCA Component Coefficients')
plt.show()

bo = np.arange(np.where(attributeNames=='BOTR0')[0][0]+1,np.where(attributeNames=='BOTR7')[0][0]+2)

for i in pcs:    
    plt.bar(bo+i*bw, V[np.where(attributeNames=='BOTR0')[0][0]:np.where(attributeNames=='BOTR7')[0][0]+1,i], width=bw)
plt.xticks(bo+bw, attributeNames[np.where(attributeNames=='BOTR0')[0][0]:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Botright: PCA Component Coefficients')
plt.show()