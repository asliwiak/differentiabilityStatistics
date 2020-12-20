#written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:39:03 2020

@author: adam
"""

import os
import matplotlib.pyplot as plt
import numpy as np

r = 38

def ddt(x, y, z, rho, sigma = 10, beta = 8/3):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return dxdt, dydt, dzdt

def step(x, y, z, r):
    dt = 0.01
    dx0, dy0, dz0 = ddt(x, y, z, r)
    x1 = x + 0.5 * dt * dx0
    y1 = y + 0.5 * dt * dy0
    z1 = z + 0.5 * dt * dz0
    dx1, dy1, dz1 = ddt(x1, y1, z1, r)
    x += dt * dx1
    y += dt * dy1
    z += dt * dz1
    return x, y, z

N = 10000000
x = 1
y = 1
z = 1
xz = list([x,z])

for i in range(0,N):
    x, y, z = step(x,y,z,r)
    xz.append(x)
    xz.append(z)
    
    
x_arr = np.array(xz[0::2])
z_arr = np.array(xz[1::2])

fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.set_xlabel(r'$x^1$',fontsize = 20)
ax.set_ylabel(r'$x^3 - 0.9\gamma$',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
# ax.plot(x_arr, z_arr - 0.9 * r, linestyle = ' ', marker = '.', markersize = 0.2 , color = 'blue',alpha=0.6)
ax.plot(x_arr, z_arr, linestyle = ' ', marker = '.', markersize = 0.2 , color = 'blue',alpha=0.6)

# plt.plot(x_arr, z_arr - 0.9 * r, linestyle = ' ', marker = '.', markersize = 0.2 , color = 'red',alpha=0.6)

    
    
