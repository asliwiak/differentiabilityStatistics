#written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:41:26 2020

@author: adam
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0,1.0,2049)
xc = x[0:2048] - (x[1]-x[0])/2

#resc = bins/traj_length
resc = 2048/441943040000

msize = 2;

g01 = np.load('grad_0.2_0.97_n.npy')
g02 = np.load('grad_0.4_0.97_n.npy')
g03 = np.load('grad_0.6_0.97_n.npy')
g04 = np.load('grad_0.8_0.97_n.npy')
g05 = np.load('grad_1.0_0.97_n.npy')
g06 = np.load('grad_1.2_0.97_n.npy')
g07 = np.load('grad_1.4_0.97_n.npy')

d01 = np.load('dens_0.2_0.97_n.npy')
d02 = np.load('dens_0.4_0.97_n.npy')
d03 = np.load('dens_0.6_0.97_n.npy')
d04 = np.load('dens_0.8_0.97_n.npy')
d05 = np.load('dens_1.0_0.97_n.npy')
d06 = np.load('dens_1.2_0.97_n.npy')
d07 = np.load('dens_1.4_0.97_n.npy')

dd01 = (d01[2:]-d01[0:-2])/(2*(x[1]-x[0]))
dd02 = (d02[2:]-d02[0:-2])/(2*(x[1]-x[0]))
dd03 = (d03[2:]-d03[0:-2])/(2*(x[1]-x[0]))
dd04 = (d04[2:]-d04[0:-2])/(2*(x[1]-x[0]))
dd05 = (d05[2:]-d05[0:-2])/(2*(x[1]-x[0]))
dd06 = (d06[2:]-d06[0:-2])/(2*(x[1]-x[0]))
dd07 = (d07[2:]-d07[0:-2])/(2*(x[1]-x[0]))


plt.plot(xc,g01*resc, label = r'$\gamma=0.2$',linewidth = 2.0, color = 'purple')
plt.plot(xc[1:-1] , dd01*resc,linestyle=' ', marker='.',markersize = msize,color='purple')
plt.plot(xc,g02*resc, label = r'$\gamma=0.4$',linewidth = 2.0, color = 'blue')
plt.plot(xc[1:-1] , dd02*resc,linestyle=' ', marker='.',markersize = msize,color='blue')
plt.plot(xc,g03*resc, label = r'$\gamma=0.6$',linewidth = 2.0, color = 'red')
plt.plot(xc[1:-1] , dd03*resc,linestyle=' ', marker='.',markersize = msize,color='red')

plt.plot(xc,g04*resc, label = r'$\gamma=0.8$',linewidth = 2.0, color = 'orange')
plt.plot(xc[1:-1] , dd04*resc,linestyle=' ', marker='.',markersize = msize,color='orange')
plt.plot(xc,d05*resc, label = r'$\gamma=1.0$',linewidth = 2.0, color = 'green')
plt.plot(xc[1:-1] , dd05*resc,linestyle=' ', marker='.',markersize = msize,color='green')
plt.plot(xc,d06*resc, label = r'$\gamma=1.2$',linewidth = 2.0, color = 'brown')
plt.plot(xc[1:-1] , dd06*resc,linestyle=' ', marker='.',markersize = msize,color='brown')

plt.legend(loc="lower left",fontsize = 12)
plt.grid(True)
plt.xlabel('x',fontsize = 15)
plt.ylabel(r'$\rho(x)g(x)=\rho^\prime(x)$', fontsize = 15)
plt.ylim((-2,2))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
# plt.savefig('hybrid_g_1.png')
