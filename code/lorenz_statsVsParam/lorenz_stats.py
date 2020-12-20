#written by Adam Sliwiak
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:18:47 2020

@author: adam
"""


from pylab import *
from numpy import *
from scipy.optimize import curve_fit


data = loadtxt('lorenz_z.txt')
gammaAll, zAll = data[:,0], data[:,1]

zmod = zAll - 1.06*gammaAll + 0.00095 * gammaAll * gammaAll


# ----------- GENERAL ---------------
plt.plot(gammaAll,zmod,linestyle=' ', marker='.',markersize = 1.,color='orange')
plt.xlabel(r'$\gamma$',fontsize = 15)
plt.ylabel(r'$\langle x^3\rangle - s(\gamma)$', fontsize = 15)
# plt.ylabel(r'$\langle x^3\rangle$', fontsize = 15)
plt.grid(True, which="both")

#-------------PARTIAL
#useful for Figure 12-13
# for (gammaL, gammaH) in [(28, 32), (36, 40), (66, 70)]:
#     sel = logical_and(gammaAll >= gammaL, gammaAll <= gammaH)
#     gamma, z = gammaAll[sel], zmod[sel]
#     gamma, ind, cnt = unique(gamma, return_inverse=True, return_counts=True)
#     zMean = zeros_like(gamma)
#     z2Mean = zeros_like(gamma)
#     add.at(zMean, ind, z)
#     add.at(z2Mean, ind, z*z)
#     zMean /= cnt
#     z2Mean /= cnt
#     zVar = (z2Mean - zMean**2).sum() / (cnt.sum() - cnt.size)
#     print(gammaL, gammaH, sqrt(zVar))
#     J = zMean - zMean[0] - (zMean[-1] - zMean[0]) / (gamma[-1] - gamma[0]) * (gamma - gamma[0])
#     # J = zMean
#     figure(figsize=(8, 8))
#     errorbar(gamma, J, 3 * sqrt(zVar), fmt='.k', ms=1)
#     xticks(fontsize=15)
#     yticks(fontsize=15)
#     yticks([min(J),max(J)])
#     gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#     xlabel(r'$\gamma$',fontsize=20)
#     ylabel(r'$\bar{\langle J\rangle}$',fontsize=20, rotation = 'horizontal')
#     savefig('lorenz_system_{}_{}.png'.format(gammaL, gammaH))
    


