#written by Qiqi Wang
from pylab import *
from numpy import *
from scipy.optimize import curve_fit

rhoAll, zAll = loadtxt('lorenz_z.txt').T
figure(1, figsize=(24,10))
rho, ind, cnt = unique(rhoAll, return_inverse=True, return_counts=True)
zMean = zeros_like(rho)
z2Mean = zeros_like(rho)
add.at(zMean, ind, zAll)
add.at(z2Mean, ind, zAll*zAll)
zMean /= cnt
z2Mean /= cnt
zVar = (z2Mean - zMean**2).sum() / (cnt.sum() - cnt.size)
# errorbar(rho, zMean - 1.06 * rho + 0.00095 * rho**2, 3 * sqrt(zVar), fmt='.b', ms=0.1)
ylim([-5.6, -5.2])
#xlabel(r'$\rho$', fontsize=25)
#ylabel(r'$\overline{z} - 1.06 \rho + 0.00095 \rho^2$', fontsize=25)
xticks(fontsize=25)
yticks(fontsize=25)

for rhoL, rhoH in [(28, 32), (36, 40), (66, 70)]:
    sel = logical_and(rhoAll >= rhoL, rhoAll <= rhoH)
    rho, z = rhoAll[sel], zAll[sel]
    rho, ind, cnt = unique(rho, return_inverse=True, return_counts=True)
    zMean = zeros_like(rho)
    z2Mean = zeros_like(rho)
    add.at(zMean, ind, z)
    add.at(z2Mean, ind, z*z)
    zMean /= cnt
    z2Mean /= cnt
    zVar = (z2Mean - zMean**2).sum() / (cnt.sum() - cnt.size)
    J = zMean - 1.06 * rho + 0.00095 * rho**2
    
    # figure(1)
    # errorbar(rho, J, 3 * sqrt(zVar), fmt='.r', ms=0.1)
    
    figure(figsize=(8, 8))
    # loglog(rho - rho[:,newaxis], abs(J - J[:,newaxis]) + 6 * sqrt(zVar), '.b', ms=1)
    loglog(rho - rho[:,newaxis], abs(J - J[:,newaxis]) - 6 * sqrt(zVar), '.k', ms=1)
    xlim([1E-3, 5])
    ymax = 10 ** ceil(log10(abs(J - J[:,newaxis]).max() + 6 * sqrt(zVar)))
    ylim([2E-4 * ymax, ymax])
    # grid()
    for i in range(9):
        plot([1E-8 * 10**i, 1E-4 * 10**i], [1E-4 * ymax, ymax], 'k', lw=0.5)
    xticks(fontsize=15)
    yticks(fontsize=15)
    xlabel(r'$|x-y|$',fontsize=20)
    ylabel(r'$|\bar{\langle J\rangle}(x)-\bar{\langle J\rangle}(y)|$',fontsize=20)
    title(r'$\gamma\in[{},{}]$'.format(rhoL,rhoH),fontsize=20)
    savefig('lorenz_Holder_{}_{}.png'.format(rhoL, rhoH))

# figure(1)
# savefig('lorenz_z.png')
