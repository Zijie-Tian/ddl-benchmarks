import torch

import timeit

def host2gpu(x, gpu_device=torch.device("cuda:0"), non_blocking=False):
    assert x.is_cpu
    return x.to(device=gpu_device, non_blocking=non_blocking)

def gpu2host(x, non_blocking=False):
    assert x.is_cuda
    return x.to(device=torch.device("cpu"), non_blocking=non_blocking)

if __name__ == '__main__':
    rank = 0
    gpu_device = torch.device("cuda:"+str(rank))
    x_gpu = torch.randn(256, 1024, 1024, dtype=torch.float32, device=gpu_device) # 1GB
    x_cpu = torch.randn(256, 1024, 1024, dtype=torch.float32, device=torch.device("cpu")) # 1GB
    x_cpu_pinned = torch.randn(256, 1024, 1024, dtype=torch.float32, device=torch.device("cpu"), pin_memory=True) # 1GB

    t0 = timeit.Timer(
        stmt="gpu2host(x, non_blocking=False)",
        setup='from __main__ import gpu2host',
        globals={
            'x' : x_gpu,
            'gpu_device' : gpu_device
        },
    )

    t1 = timeit.Timer(
        stmt="gpu2host(x, non_blocking=True)",
        setup='from __main__ import gpu2host',
        globals={
            'x' : x_gpu,
            'gpu_device' : gpu_device
        },
    )

    torch.cuda.synchronize(gpu_device)

    t2 = timeit.Timer(
        stmt="host2gpu(x, gpu_device=gpu_device, non_blocking=False)",
        setup='from __main__ import host2gpu',
        globals={
            'x' : x_cpu,
            'gpu_device' : gpu_device
        },
    )

    t3 = timeit.Timer(
        stmt="host2gpu(x, gpu_device=gpu_device, non_blocking=False)",
        setup='from __main__ import host2gpu',
        globals={
            'x' : x_cpu_pinned,
            'gpu_device' : gpu_device
        },
    )

    print(f'GPU to CPU bandwidth  {1 / (t0.timeit(10) / 10) :>5.1f} GB / s')
    print(f'non-blocking GPU to CPU bandwidth  {1 / (t1.timeit(10) / 10) :>5.1f} GB / s')

    print(f'CPU to GPU bandwidth  {1 / (t2.timeit(10) / 10) :>5.1f} GB / s')
    print(f'Pinned CPU to GPU bandwidth  {1 / (t3.timeit(10) / 10) :>5.1f} GB / s')



