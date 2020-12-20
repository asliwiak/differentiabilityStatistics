#written by Adam Sliwiak
from pylab import *
from numpy import *

figure(figsize=(20,10))
for fname in ['ntail_1.3_0.97.npy','ntail_1.1_0.97.npy', 'ntail_0.9_0.97.npy', 'ntail_0.7_0.97.npy', 'ntail_0.5_0.97.npy', 'ntail_0.3_0.97.npy']:
    g = load(fname)
    bins = 10**(arange(2049) / 20 - 18)
    hist((bins[1:] + bins[:-1]) / 2, bins=bins, weights=g, alpha=0.9)
xscale('log')
yscale('log')
axis('scaled')
xlim([1E-0, 1E13])
gca().set_xticks(logspace(-4,13,18))
for label in gca().ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ylim([0.1, 1E10])
gca().set_yticks(logspace(0,9,10))
grid()
xlabel(r'$|g|$',fontsize = 20)
ylabel('Number of appearances',fontsize = 20)
for tick in gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(20)

legend([
         '$\gamma=1.3$',
         '$\gamma=1.1$',
         '$\gamma=0.9$',
         '$\gamma=0.7$',
         '$\gamma=0.5$',
         '$\gamma=0.3$'], fontsize=20)


