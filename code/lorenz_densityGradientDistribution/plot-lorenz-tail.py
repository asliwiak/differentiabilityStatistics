#written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:18:56 2020

@author: adam
"""


import os
from pylab import *
from numpy import *

rhos = {68: 'b', 40: 'g', 28: 'r'}
# rhos = [68, 38, 28]
# rhos = {40: 'r'}
figure(figsize=(20,10))
# for rho, color in rhos.items():

for rho in rhos:
    g = 0
    for proc in [5,1,2,3,4,6]:
        fname = 'g_gathered_proc{}_{}.npy'.format(proc, rho)
        if not os.path.exists(fname):
            break
        g = g + load(fname)
    bins = 10**(arange(2049) / 20 - 18)
    hist((bins[1:] + bins[:-1]) / 2, bins=bins, weights=g, alpha=0.9)
    # plot([bins[:-1], bins[1:]], [g, g], color)

xscale('log')
yscale('log')
axis('scaled')
xlim([1E-8, 1E10])
gca().set_xticks(logspace(-7,9,17))
ylim([0.1, 1E10])
gca().set_yticks(logspace(0,9,10))
grid()
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
    
xlabel(r'$|g|$',fontsize = 20)
ylabel('Number of appearances',fontsize = 20)

legend(['$\gamma=68$',
        '$\gamma=40$',
        '$\gamma=28$'], fontsize=20)

#line
# xgrid = logspace(-2,7,num = 100, base = 10)
# loglog(xgrid,1e+6*xgrid**(-1),linestyle='--')


savefig('lorenz_g_tail.png')
