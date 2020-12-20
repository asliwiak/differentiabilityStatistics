#written by Qiqi Wang
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

dataAll = loadtxt('onion.97.txt')
gammaAll, zAll = dataAll[:,0], dataAll[:,2:4]
gammaRanges = [(.2, .45), (.65, .9), (1.1, 1.35), (1.55, 1.8)]
Jranges = [(.2, .45), (.65, .9), (1.1, 1.35), (1.55, 1.8), (.2, .45), (.65, .9), (1.1, 1.35), (1.55, 1.8)]
ymax = zeros(len(gammaRanges))
kk = 0
for i in range(zAll.shape[1]):
    for iplot, (gammaL, gammaH) in enumerate(gammaRanges):
        sel = logical_and(gammaAll >= gammaL, gammaAll <= gammaH)
        gamma, z = gammaAll[sel], zAll[sel,i]
        gamma, ind, cnt = unique(gamma, return_inverse=True, return_counts=True)
        zMean = zeros_like(gamma)
        z2Mean = zeros_like(gamma)
        add.at(zMean, ind, z)
        add.at(z2Mean, ind, z*z)
        zMean /= cnt
        z2Mean /= cnt
        zVar = (z2Mean - zMean**2).sum() / (cnt.sum() - cnt.size)
        print(gammaL, gammaH, sqrt(zVar))

        J = zMean - zMean[0] - (zMean[-1] - zMean[0]) / (gamma[-1] - gamma[0]) * (gamma - gamma[0])
        Jranges[kk] = (min(J),max(J))
        kk = kk + 1
        
        figure(iplot + 11, figsize=(8, 8))
        errorbar(gamma, J, 3 * sqrt(zVar), fmt='.k', ms=1)
        
        figure(iplot+1, figsize=(8, 8))
        loglog(gamma - gamma[:,newaxis], abs(J - J[:,newaxis]) - 6 * sqrt(zVar), '.k', ms=0.1)
        ymax[iplot] = max(ymax[iplot], 10 ** ceil(log10(abs(J - J[:,newaxis]).max() + 6 * sqrt(zVar))))

kk = 0
for iplot, (gammaL, gammaH) in enumerate(gammaRanges):
    figure(iplot+1)
    xlim([5E-5, 0.25])
    ylim([2E-4 * ymax[iplot], ymax[iplot]])
    for i in range(9):
        plot([1E-8 * 10**i, 1E-4 * 10**i], [1E-4 * ymax[iplot], ymax[iplot]], 'k', lw=0.5)
    xticks(fontsize=15)
    yticks(fontsize=15)
    title(r'$\gamma\in[{},{}]$'.format(gammaL,gammaH),fontsize=15)
    xlabel(r'$|x-y|$',fontsize=20)
    ylabel(r'$|\bar{\langle J\rangle}(x)-\bar{\langle J\rangle}(y)|$',fontsize=20)
    savefig('onion_Holder_{}_{}.png'.format(gammaL, gammaH))

    figure(iplot+11)
    xticks(fontsize=15)
    yticks(fontsize=15)
    yticks((min(Jranges[kk][0],Jranges[kk+4][0]),max(Jranges[kk][1],Jranges[kk+4][1])))
    gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    kk = kk + 1
    xlabel(r'$\gamma$',fontsize=20)
    ylabel(r'$\bar{\langle J\rangle}$',fontsize=20,rotation='horizontal')
    savefig('onion_map_{}_{}.png'.format(gammaL, gammaH))
