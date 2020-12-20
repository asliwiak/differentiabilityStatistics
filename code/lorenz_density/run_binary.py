#written by Qiqi Wang
# density/run_binary.py
import os
import sys
import json
import subprocess
from numpy import *

binary =  'your-path-to-binary-here'
#params = [sigma, gamma, beta, time step, N of time steps]
params = [10, 70, 2.666667, 0.002, 5000]

for iIter in range(1):
    print('iIter=', iIter, flush=True)
    procs = []
    gpus = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    numGpus = len(gpus.strip().splitlines())
    for i in [0]:
        p = subprocess.Popen(binary, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.stdin.write(array([i, random.randint(1 << 30)], uint32).tobytes())
        p.stdin.write(array(params, float32).tobytes())
        p.stdin.flush()
        procs.append(p)
    
    for p in procs:
        p.wait()
