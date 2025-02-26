# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:28:50 2025

@author: gusta
"""

import numpy as np
import matplotlib.pyplot as plt

mean, sigma = 10, 1

R_gauss = np.random.normal(mean, sigma, 1000)

print('Distribución Gaussiana\n')
print(np.max(R_gauss), np.min(R_gauss))

counts_gauss, bins_gauss = np.histogram(R_gauss)

print(counts_gauss, bins_gauss)
print('-'*100)

plt.figure(1)
plt.hist(bins_gauss[:-1], bins_gauss, weights=counts_gauss)
plt.show()
print('Distribución Lognormal\n')
R_lognormal = np.random.lognormal(mean, sigma, 1000)

print(np.max(R_lognormal), np.min(R_lognormal))

counts_lognormal, bins_lognormal = np.histogram(R_lognormal)

print(counts_lognormal, bins_lognormal)
print('-'*100)
plt.figure(2)
plt.hist(bins_lognormal[:-1], bins_lognormal, weights=counts_lognormal)
plt.show()

