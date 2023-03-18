import os
import sys
import random
import logging
from threading import Lock
import argparse
import timeit
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import models, datasets, transforms

from ptflops.utils import flops_to_string, params_to_string
from ptflops.pytorch_engine import add_flops_counting_methods, CUSTOM_MODULES_MAPPING

logging.basicConfig(level=logging.INFO)

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
master_addr = os.environ['MASTER_ADDR']
master_port = os.environ['MASTER_PORT']

def log(s, nl=True):
    if local_rank != 0:
        return
    print(s, end='\n' if nl else '')
    sys.stdout.flush()

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=100,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--flops', action='store_true', default=False,
                    help='enable flops profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')

# set args as global var.
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# --------- Parameter Server --------------------
class ParameterServer(object):
    def __init__(self, args, num_gpus=0):
        super().__init__()
        self.lock = Lock()
        self.future_model = torch.futures.Future()
        model = getattr(models, args.model)(num_classes=args.num_classes)
        
        self.model = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
    
    def get_model(self):
        return self.model
    
    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, worker_rank):
        self = ps_rref.local_value()
        with self.lock:
            # print(f'PS updates parameters based on gradients from worker{worker_rank}')
            # update model parameters
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g
            self.optimizer.step()
            self.optimizer.zero_grad()

            fut = self.future_model

            fut.set_result(self.model)
            self.future_model = torch.futures.Future()

        return fut

# --------- Generate Fake Datasets --------------------
# Set up fake data
if local_rank < world_size - 1:
    datasets = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224)
        target = torch.LongTensor(args.batch_size).random_() % 1000
        if args.cuda:
            data = data.to(local_rank)
            target = target.to(local_rank)
        datasets.append(data)
    data_index = 0

def _run_trainer(ps_rref, rank):
    """
        Note : this rank is not dist env 'local_rank', it is the rank in trainer processes.
    """
    global data_index

    model = ps_rref.rpc_sync().get_model().to(rank)

    if args.flops:
            model = add_flops_counting_methods(model)

    img_secs = []
    time_sum = 0
    ngpu = world_size - 1

    # Warmup
    for _ in range(args.num_batches_per_iter):
        with dist_autograd.context() as context_id:
            data = datasets[data_index%len(datasets)]
            data_index += 1
            # opt.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            # loss.backward()
            dist_autograd.backward(context_id, [loss])
    
    # Run training loop.
    for iter in range(args.num_iters):
        start = time.perf_counter() 
        for _ in range(args.num_batches_per_iter):
            '''
                use Autograd will be much faster !!!!
            '''
            with dist_autograd.context() as context_id:
                data = datasets[data_index%len(datasets)]
                data_index += 1
                # opt.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                # loss.backward()
                dist_autograd.backward(context_id, [loss])

            model = rpc.rpc_sync(to=ps_rref.owner(),
                        func=ParameterServer.update_and_fetch_model,
                        args=(ps_rref, [p.grad for p in model.cpu().parameters()], rank)
                        ).to(rank)

        end = time.perf_counter()
        time_sum += (end - start)

        img_sec = args.batch_size * args.num_batches_per_iter / (end - start)
        log('Iter #%d: %.1f img/sec per %s' % (iter, img_sec, "GPU"))
        img_secs.append(img_sec)

    if args.flops:
        flops_count, params_count = model.compute_average_flops_cost()
        flops_count *= model.__batch_counter__
        model.stop_flops_count()

    if args.flops:
        log("Flops: {} GFlops\n".format(flops_count / (1000 ** 3) / time_sum))
 
    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log("Avg Time: {} sec\n".format(time_sum / args.num_iters))
    log('Img/sec per %s: %.1f +-%.1f' % (ngpu, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (ngpu, "GPU", ngpu * img_sec_mean, ngpu * img_sec_conf))

if __name__ == "__main__":
    if local_rank == world_size - 1:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug("local_rank: {}, world_size: {}".format(local_rank, world_size))
    logging.debug("master addr: {}, master port: {}".format(master_addr, master_port))
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "env://"

    if local_rank == world_size - 1:
        rpc.init_rpc(
            "parameter_server",
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        ps_rref = rpc.RRef(ParameterServer(args, num_gpus=world_size - 1))

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in range(world_size - 1):
            trainer_name = "trainer{}".format(trainer_rank)
            logging.debug("[cmd] RPC Call {} to run trainer".format(trainer_name))
            fut = rpc.rpc_async(
                to=trainer_name, 
                func=_run_trainer, 
                args=(ps_rref, trainer_rank,)
            )
            futs.append(fut)

        logging.debug("Waiting for all trainers to finish")
        torch.futures.wait_all(futs)
        logging.info("PS EXIT")
    
    else :
        # Initialize RPC.
        trainer_name = "trainer{}".format(local_rank)
        rpc.init_rpc(
            trainer_name,
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        logging.debug("Trainer {} is running".format(local_rank))
        # Trainer just waits for RPCs from master.

    # block until all rpcs finish
    rpc.shutdown()