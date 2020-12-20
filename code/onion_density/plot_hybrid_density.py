#written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:41:26 2020

@author: adam
"""


import numpy as np
import matplotlib.pyplot as plt


d01 = np.load('dens_0.2_0.97_n.npy')
d02 = np.load('dens_0.4_0.97_n.npy')
d03 = np.load('dens_0.6_0.97_n.npy')
d04 = np.load('dens_0.8_0.97_n.npy')
d05 = np.load('dens_1.0_0.97_n.npy')
d06 = np.load('dens_1.2_0.97_n.npy')
d07 = np.load('dens_1.4_0.97_n.npy')


x = np.linspace(0.0,1.0,2049)
xc = x[0:2048] - (x[1]-x[0])/2

resc = 2048/sum(d01)

plt.plot(xc,d01*resc, label = r'$\gamma=0.2$',linewidth = 2.0, color = 'purple')
plt.plot(xc,d02*resc, label = r'$\gamma=0.4$',linewidth = 2.0, color = 'blue')
plt.plot(xc,d03*resc, label = r'$\gamma=0.6$',linewidth = 2.0, color = 'red')
plt.plot(xc,d04*resc, label = r'$\gamma=0.8$',linewidth = 2.0, color = 'orange')
plt.plot(xc,d05*resc, label = r'$\gamma=1.0$',linewidth = 2.0, color = 'green')
plt.plot(xc,d06*resc, label = r'$\gamma=1.2$',linewidth = 2.0, color = 'brown')
plt.plot(xc,d07*resc, label = r'$\gamma=1.4$',linewidth = 2.0, color = 'grey')
# plt.plot(xc,d05*resc, label = r'$\gamma=2.2$',linewidth = 2.0, color = 'black')
plt.legend(loc="upper left",fontsize = 12)
plt.grid(True)
plt.xlabel('x',fontsize = 15)
plt.ylabel(r'$\rho(x)$', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylim((-0.2,5))
# plt.savefig('hybrid_density_2.png')
