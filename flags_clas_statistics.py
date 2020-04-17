# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:35:02 2020

@author: larsh
"""
import scipy.stats as st
from scipy.stats import norm
import numpy as np

Error_test_ANN = np.array([0.50, 0.70, 0.60, 0.60, 0.63, 0.52, 0.63, 0.42, 0.73, 0.68])
Error_test_rlr = np.array([0.30, 0.60, 0.55, 0.35, 0.52, 0.57, 0.57, 0.52, 0.42, 0.47])
Error_test     = np.array([0.39, 0.20, 0.45, 0.30, 0.31, 0.15, 0.31, 0.52, 0.21, 0.31])

dash = '-' * 80
# STATISTICS

alpha = 0.05

Lowerconf=np.zeros(3)
Upperconf=np.zeros(3)
pvalue=np.zeros(3)

# ANN vs. RLR
z = Error_test_ANN-Error_test_rlr
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z)) # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1) # p-value
Lowerconf[0] = CI[0]
Upperconf[0] = CI[1]
pvalue[0] = p

# ANN vs baseline
z = Error_test_ANN-Error_test
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z)) # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1) # p-value
Lowerconf[1] = CI[0]
Upperconf[1] = CI[1]
pvalue[1] = p

# RLR vs baseline
z = Error_test_rlr-Error_test
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z)) # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1) # p-value
Lowerconf[2] = CI[0]
Upperconf[2] = CI[1]
pvalue[2] = p


print(dash)
print("{:>15s}{:>15s}{:>15s}{:>15s}".format("","ANN-RLR","ANN-Base","Base-RLR"))
print("{:>15s}{:>15f}{:>15f}{:>15f}".format("Lowerconf",Lowerconf[0],Lowerconf[1],Lowerconf[2]))
print("{:>15s}{:>15f}{:>15f}{:>15f}".format("Upperconf",Upperconf[0],Upperconf[1],Upperconf[2]))
print("{:>15s}{:>15f}{:>15f}{:>15f}".format("pvalue",pvalue[0],pvalue[1],pvalue[2]))
print(dash)