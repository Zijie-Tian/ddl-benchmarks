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

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size

def host2gpu(x, gpu_device=torch.device("cuda:0"), non_blocking=False):
    # assert x.is_cpu
    return x.to(device=gpu_device, non_blocking=non_blocking)

def gpu2host(x, non_blocking=True):
    assert x.is_cuda
    return x.to(device=torch.device("cpu"), non_blocking=non_blocking)

if __name__ == '__main__':
    model = torchvision.models.resnet50(num_classes=30).cpu()

    t0 = timeit.Timer(
        stmt="host2gpu(x, gpu_device=gpu_device, non_blocking=False)",
        setup='from __main__ import host2gpu',
        globals={
            'x' : model,
            'gpu_device' : torch.device("cuda:0")
        },
    )

    model_size = get_model_size(model)

    print(f'CPU to GPU bandwidth  {model_size / 1024**3 / (t0.timeit(10) / 10) :>5.1f} GB / s')