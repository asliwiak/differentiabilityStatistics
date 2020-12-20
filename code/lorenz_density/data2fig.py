#written by Qiqi Wang
import os
import multiprocessing
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import *

def process(i):
    if os.path.exists('figs/step_{:04d}.png'.format(i)):
        return
    print(i, flush=True)
    a = open('data/device_0_step_{}.bin'.format(i), 'rb').read()
    a = frombuffer(a, dtype=float32).reshape([3840,2160]).copy()
    a[a == 0] = 0.8
    amax = max(min(1E8,
               1E8 * exp(-log(1E8 / 1E4) / 2000 * (i - 500))),
               1E4 * exp(-log(1E4 / 7E3) / 2500 * (i - 2500)))
    fig = figure(figsize=(38.4, 21.6))
    ax = subplot(1,1,1)
    im = imshow(a.T, extent=[-20,20,0,50], aspect='auto',
           norm=LogNorm(vmin=0.8, vmax=amax), cmap='inferno', origin='lower')
    subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    cbaxes = ax.inset_axes([0.9, 0.05, 0.03, 0.4])
    ticks = 10**arange(floor(log10(amax)) + 1)
    cbar = colorbar(im, ticks=ticks, cax=cbaxes)
    cbar.ax.tick_params(labelsize=50, color='white', labelcolor='white')
    ax.text(.5,.95,'t={:5.2f}'.format((i+1)*0.01),
            horizontalalignment='center', fontsize=50,
            transform=ax.transAxes, color='white')

    savefig('figs/step_{:04d}.png'.format(i), dpi=100)
    close(fig)

p = multiprocessing.Pool()
p.map(process, range(0, 5000, 1))
