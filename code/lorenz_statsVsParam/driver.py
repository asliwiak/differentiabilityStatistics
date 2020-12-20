#written by Qiqi Wang
import os
from numpy import linspace

for rho in linspace(29, 30, 1001)[1:-1]:
    os.system('python3 lorenz.py {} >> z29.txt'.format(rho))
