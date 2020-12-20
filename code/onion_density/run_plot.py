#written by Qiqi Wang
import os
import sys
import subprocess
from numpy import *

binary = "/home/asliwiak/density/hybrid_density"

#specify parameters [gamma,h]
params = [1.3, 0.97]
procs = []
#change nIter to manipulate trajectory length
nIter = 256 * 3
gpus = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
numGpus = len(gpus.strip().splitlines())
for i in range(numGpus):
    p = subprocess.Popen(binary, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(array([i, random.randint(1 << 30)], uint32).tobytes())
    p.stdin.write(array(params, float32).tobytes())
    p.stdin.write(array(nIter, uint32).tobytes())
    p.stdin.flush()
    procs.append(p)

n = 2048
out = [frombuffer(p.stdout.read(n * 8), double) for p in procs]
histogram = sum(out, 0)
print('Used {} samples.'.format(sum(histogram)))
myname = subprocess.check_output('hostname').decode().strip()
params = ('_'.join(['{}'] * len(params))).format(*tuple(params))
fname = '{}_{}_{}_{}_{}.npy'.format(os.path.basename(binary),
        params, myname, nIter, random.rand())

save(fname, histogram)
