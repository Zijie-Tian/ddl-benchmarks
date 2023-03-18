import os
import argparse
import threading
from datetime import datetime
import time
import timeit
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim

import torchvision

def gpu2host(x, non_blocking=True):
    assert x.is_cuda
    return x.to(device=torch.device("cpu"), non_blocking=non_blocking)

def benchmark(msg_size):
    '''
        msg_size : in KB
    '''
    x_gpu = torch.randn(msg_size, 256, dtype=torch.float32, device=torch.device(0)) # msg_size KB
    t0 = timeit.Timer(
        stmt="gpu2host(x, non_blocking=False)",
        setup='from __main__ import gpu2host',
        globals={
            'x' : x_gpu,
        }
    )

    bandwidth = msg_size / 1024 ** 2 / (t0.timeit(10) / 10)

    print(f'{msg_size}KB GPU to CPU bandwidth : { bandwidth :>5.1f} GB / s')

    return bandwidth

if __name__ == '__main__':
    bandwidth_arr = []
    size_arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for size in size_arr:
        bandwidth_arr.append(benchmark(size))

    for size in size_arr:
        bandwidth_arr.append(benchmark(size * 1024))

    import numpy as np
    ret = np.asarray([size_arr + [s * 1024 for s in size_arr], bandwidth_arr])

    np.savetxt('d2h_bandwidth.csv', ret.T, delimiter=',')
    
    
