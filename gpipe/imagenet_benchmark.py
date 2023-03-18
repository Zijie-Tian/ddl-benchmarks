"""ResNet-101 Speed Benchmark"""
import argparse
import platform
import time
import timeit
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD

from resnet import resnet101

import sys
sys.path.append('../3rdparty/torchgpipe')
import torchgpipe
from torchgpipe import GPipe

sys.path.append('../3rdparty/ptflops')
from ptflops.utils import flops_to_string, params_to_string
from ptflops.pytorch_engine import add_flops_counting_methods, CUSTOM_MODULES_MAPPING

logging.basicConfig(level=logging.INFO)

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')

parser.add_argument('--chunks', type=int, default=8,
                    help='number of gpipe pipeline chunks.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--partitions', type=int, default=4, choices=[1, 2, 4, 8],
                    help='partition size')

parser.add_argument('--num-warmup-batches', type=int, default=5,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--flops', action='store_true', default=False,
                    help='enable flops profiler')

args = parser.parse_args()

balance_setup = {
    1   :   [370],
    2   :   [135, 235],
    4   :   [44, 92, 124, 110],
    8   :   [26, 22, 33, 44, 44, 66, 66, 69]
}

devices=[
    'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3',
    'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'
]

model: nn.Module = resnet101(num_classes=args.num_classes)
model = cast(nn.Sequential, model)
model = GPipe(model, balance_setup[args.partitions], devices=devices, chunks=args.chunks)

if args.flops:
    model = add_flops_counting_methods(model)

optimizer = SGD(model.parameters(), lr=0.1)

def benchmark_step(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step() 

if args.flops:
        model.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])
    
time_sum = 0
img_secs = []
data = torch.rand(args.batch_size, 3, 224, 224).to(model.devices[0])
target = torch.LongTensor(args.batch_size).random_().to(model.devices[-1]) % 1000

timer = timeit.Timer(
    stmt='benchmark_step(data, target)', 
    setup='from __main__ import benchmark_step',
    globals={'data' : data, 'target' : target}
)

# Warmup
t = timer.timeit(args.num_warmup_batches)

for x in range(args.num_iters):
    t = timer.timeit(args.num_batches_per_iter)
    time_sum += t

    img_sec = args.batch_size * args.num_batches_per_iter / t
    logging.info('Iter #%d: %.1f img/sec ' % (x, img_sec))
    img_secs.append(img_sec)

if args.flops:
    flops_count, params_count = model.compute_average_flops_cost()
    flops_count *= model.__batch_counter__
    model.stop_flops_count()

if args.flops:
    logging.info("Flops: {} GFlops\n".format(flops_count / (1000 ** 3) / time_sum))
   
# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
logging.info("Avg Time per step : {} sec\n".format(time_sum / args.num_iters))
logging.info('Img/sec : %.1f +-%.1f' % (img_sec_mean, img_sec_conf))
logging.info('Total img/sec on %d GPU(s): %.1f +-%.1f' %
    (len(model.devices), img_sec_mean, img_sec_conf))





