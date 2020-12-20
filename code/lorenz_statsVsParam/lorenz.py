#written by Qiqi Wang
import os
import sys
from pylab import *
import matplotlib.colors as colors
from numba import cuda

sigma, rho, beta = 10, float(sys.argv[1]), 8./3

@cuda.jit(device=True)
def ddt(x, y, z):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return dxdt, dydt, dzdt

@cuda.jit(device=True)
def step(x, y, z):
    dt = 0.01
    dx0, dy0, dz0 = ddt(x, y, z)
    x1 = x + 0.5 * dt * dx0
    y1 = y + 0.5 * dt * dy0
    z1 = z + 0.5 * dt * dz0
    dx1, dy1, dz1 = ddt(x1, y1, z1)
    x += dt * dx1
    y += dt * dy1
    z += dt * dz1
    return x, y, z

@cuda.jit
def step_all(zSum, xyz, nsteps):
    gid = cuda.grid(1)
    if gid < xyz.shape[0]:
        for i in range(nsteps):
            x, y, z = xyz[gid]
            x, y, z = step(x, y, z)
            xyz[gid,:] = x, y, z
            zSum[gid] += z

if __name__ == '__main__':
    N = 10000
    xyz = rand(N, 3) * 3 + 20
    xyz_d = cuda.to_device(xyz)

    itempergroup = 32
    groupperrange = (xyz.size + (itempergroup - 1)) // itempergroup

    for i in range(10):
        zSum_d = cuda.to_device(zeros(N))
        step_all[groupperrange, itempergroup](zSum_d, xyz_d, 10000)

        zSum_d = cuda.to_device(zeros(N))
        nsteps = 1000000
        step_all[groupperrange, itempergroup](zSum_d, xyz_d, nsteps)
        print(rho, (zSum_d.copy_to_host() / nsteps).mean(), flush=True)
