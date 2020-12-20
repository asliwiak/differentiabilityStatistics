#written by Qiqi Wang
import os
import sys
from pylab import *
from autograd.numpy import *
from autograd import *
from stl import mesh
import matplotlib.colors as colors

sigma, rho, beta = 10, 28, 8./3

def ddt(xyz):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return array([dxdt, dydt, dzdt])

def step(xyz):
    dt = 0.01
    xyz1 = xyz + 0.5 * dt * ddt(xyz)
    return xyz + dt * ddt(xyz1)

stepJacobian = make_jvp(step)

def stepV(xyzv):
    xyz0 = xyzv[:3]
    v0 = xyzv[3:6]
    v1 = xyzv[6:9]
    xyz, v0 = stepJacobian(xyz0)(v0)
    xyz, v1 = stepJacobian(xyz0)(v1)
    v0 /= sqrt((v0**2).sum(0))
    v1 -= v0 * (v0 * v1).sum(0)
    v1 /= sqrt((v1**2).sum(0))
    return concatenate([xyz, v0, v1], axis=0)

def make_hessian(func):
    funcJacobian = make_jvp(func)
    def firstCall(x0):
        def secondCall(dx0):
            dfunc = lambda x : funcJacobian(x)(dx0)[1]
            ddfunc = make_jvp(dfunc)(x0)
            return lambda dx1 : ddfunc(dx1)[1]
        return secondCall
    return firstCall

stepHessian = make_hessian(step)

def stepVW(xyzvw):
    xyz0 = xyzvw[:3]
    v0 = xyzvw[3:6]
    v1 = xyzvw[6:9]
    xyz, q0 = stepJacobian(xyz0)(v0)
    xyz, q1 = stepJacobian(xyz0)(v1)
    # Orthonormalization
    R00 = sqrt((q0**2).sum(0))
    q0 /= R00
    R01 = (q0 * q1).sum(0)
    q1 -= q0 * R01
    R11 = sqrt((q1**2).sum(0))
    q1 /= R11
    # Curvature
    w00 = xyzvw[9:12]
    w01 = xyzvw[12:15]
    w11 = xyzvw[15:18]
    w00 = stepJacobian(xyz0)(w00)[1] + stepHessian(xyz0)(v0)(v0)
    w01 = stepJacobian(xyz0)(w01)[1] + stepHessian(xyz0)(v0)(v1)
    w11 = stepJacobian(xyz0)(w11)[1] + stepHessian(xyz0)(v1)(v1)
    w11 = (w11 - 2 * w01 * R01 / R00 + w00 * R01**2 / R00**2) / R11**2
    w01 = (w01 - w00 * R01 / R00) / (R00 * R11)
    w00 = w00 / R00**2
    return concatenate([xyz, q0, q1, w00, w01, w11], axis=0)

if __name__ == '__main__':
    n = 101
    steps = 5000
    xyz = array([rand(n), zeros(n), ones(n)])
    v0 = array([ones(n), zeros(n), zeros(n)])
    v1 = array([zeros(n), zeros(n), ones(n)])
    w00, w01, w11 = zeros([3,3,n])
    xyzvw = concatenate([xyz, v0, v1, w00, w01, w11], axis=0)
    for iBatch in range(1000):
        g = []
        for i in range(5000):
            xyzvw = stepVW(xyzvw)
            v0 = xyzvw[3:6]
            w00 = xyzvw[9:12]
            g.append((v0 * w00).sum(0))

        bins = 10**(arange(2049) / 20 - 18)
        h,_ = histogram(abs(ravel(g)), bins)
        save('g_{}_{}.npy'.format(rho, iBatch), h)
