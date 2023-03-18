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

import logging

logging.basicConfig(level=logging.INFO)

# batch_size = 32

# args.num_classes = 30
# batch_update_size = 1
# num_batches = 7

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-gpus', type=int, default=1,
                    help='# of GPU')
parser.add_argument('--num-warmup-batches', type=int, default=10,
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
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')
args = parser.parse_args()

image_w = 224
image_h = 224

class BatchUpdateParameterServer(object):
    def __init__(self, batch_update_size=1):
        # This model is on CPU
        self.model = torchvision.models.resnet50(num_classes=args.num_classes).cpu()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        logging.debug(f"PS got {self.curr_update_size}/{self.batch_update_size} updates")
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                logging.debug("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer(object):
    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(args.batch_size) \
                                    .random_(0, args.num_classes) \
                                    .view(args.batch_size, 1)
        self.iter_cnt = 0

    def get_next_batch(self, device=torch.device("cuda:0")):
        for _ in range(args.num_iters + args.num_warmup_batches):
            inputs = torch.randn(args.batch_size, 3, image_w, image_h)
            labels = torch.zeros(args.batch_size, args.num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
            self.iter_cnt += 1
            yield inputs.to(device=device), labels.to(device=device)

    def train(self):
        name = rpc.get_worker_info().name
        id = rpc.get_worker_info().id
        m = self.ps_rref.rpc_sync().get_model().cuda(torch.device(id - 1))
        img_secs = []
        for inputs, labels in self.get_next_batch(device=torch.device(id - 1)):
            logging.debug(f"{name} processing one batch")
            tic = time.time()
            self.loss_fn(m(inputs), labels).backward()

            if self.iter_cnt < args.num_warmup_batches:
                continue
            
            logging.debug(f"{name} reporting grads")

            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).cuda(torch.device(id - 1))
            toc = time.time()
            logging.debug(f"{name} got updated model")

            img_sec = args.batch_size / (toc - tic)
            if id == 1:
                logging.info('Iter #%d: %.1f img/sec per %s' % (self.iter_cnt - args.num_warmup_batches + 1, img_sec, "GPU"))
                img_secs.append(img_sec)

        if id == 1:
            logging.info('Avg img/sec: %.1f +- %.1f' % (np.mean(img_secs), 1.96 * np.std(img_secs)))


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()

def run_ps(trainers):
    logging.debug("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer(batch_update_size=args.num_gpus))
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    logging.debug("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = args.num_gpus + 1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)