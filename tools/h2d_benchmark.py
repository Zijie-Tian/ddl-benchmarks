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

def host2gpu(x, gpu_device=torch.device("cuda:0"), non_blocking=False):
    assert x.is_cpu
    return x.to(device=gpu_device, non_blocking=non_blocking)

def benchmark(msg_size):
    '''
        msg_size : in KB
    '''
    # x_gpu = torch.randn(msg_size, 256, dtype=torch.float32, device=torch.device(0)) # msg_size KB
    x_cpu = torch.randn(msg_size, 256, dtype=torch.float32, device=torch.device("cpu"), pin_memory=True) # 1GB
    t0 = timeit.Timer(
        stmt="host2gpu(x, non_blocking=False)",
        setup='from __main__ import host2gpu',
        globals={
            'x' : x_cpu,
        }
    )

    bandwidth = msg_size / 1024 ** 2 / (t0.timeit(10) / 10)
    print(f'{msg_size}KB CPU to GPU bandwidth : { bandwidth :>5.1f} GB / s')

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

    np.savetxt('h2d_bandwidth.csv', ret.T, delimiter=',')
    

