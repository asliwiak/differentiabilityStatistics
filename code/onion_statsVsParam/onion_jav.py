//written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:07:25 2020

@author: adam
"""


import numpy as np
import matplotlib.pyplot as plt

dataAll = np.loadtxt('onion.97.txt')
gammaAll, zAll = dataAll[:,0], dataAll[:,2:4]

plt.plot(gammaAll,zAll[:,0],linestyle=' ', marker='.',markersize = 1.,color='orange', label = r'$c=0.375,\;\epsilon = 0.125$')
plt.plot(gammaAll,zAll[:,1],linestyle=' ', marker='.',markersize = 1.,color='blue', label = r'$c=0.625,\;\epsilon = 0.125$')
plt.grid()
plt.xlabel(r'$\gamma$',fontsize = 15)
plt.ylabel(r'$\langle J\rangle$',  fontsize = 15, rotation = 'vertical')
plt.legend(loc="upper center",fontsize = 12, markerscale = 10)
