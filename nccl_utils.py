import re
import glob
import argparse
import json
import os
import heapq
import pprint

import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='nccl profile prefix (equals to directory name)', required=True)
    parser.add_argument('-t', '--target', help='target file name', required=True)
    return parser.parse_args()

def allreduce_cdf(num_devices, id2device, filepath_prefix="comscribe_reducescatter_*.csv"):
    file_paths = glob.glob(filepath_prefix)

    size_array = []

    for file_path in file_paths:
        with open(file_path) as fp:
            lines = fp.readlines()

            for line in lines:
                nodeName, commId, deviceId, src, dst, size, algo = line.split(",")
                heapq.heappush(size_array, int(size))
                size_array.append(int(size))

    i = 0
    step = 50000
    ret = {}
    while 0 < len(size_array):
        top = heapq.heappop(size_array)
        
        i = top // step * step
        ret[i] = ret.get(i, 0) + 1
        
    return ret

'''
    Do reduce on nccl results.
'''
def reduce(num_devices, id2device, filepath_prefix="comscribe_*_*.csv"):
    file_paths = glob.glob(filepath_prefix)

    # num_bytes_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
    # num_times_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
    num_bytes_comm_matrix = {}
    num_times_comm_matrix = {}
    
    for file_path in file_paths:
        nccl_type = file_path.split('_')[-2]
        if nccl_type not in num_bytes_comm_matrix.keys():
            num_bytes_comm_matrix[nccl_type] = [[0] * num_devices for _ in range(num_devices)]
            num_times_comm_matrix[nccl_type] = [[0] * num_devices for _ in range(num_devices)]

        with open(file_path) as fp:
            lines = fp.readlines()

            n_line = 0
            for line in lines:
                # print(line.split(','))
                n_line += 1
                if nccl_type == "reduce":
                    # Reduce no other Algo
                    nodeName, commId, deviceId, src, dst, size = line.split(",")
                else :
                    nodeName, commId, deviceId, src, dst, size, algo = line.split(",")
                    # print(file_path, " Line number is ", n_line)

                num_bytes_comm_matrix[nccl_type][id2device[int(dst)]][id2device[int(src)]] += int(size)
                num_times_comm_matrix[nccl_type][id2device[int(dst)]][id2device[int(src)]] += 1


    ret = {}
    for nccl_type in num_bytes_comm_matrix.keys():
        if nccl_type not in ret.keys():
            ret[nccl_type] = []

        for src in range(num_devices):
            for dst in range(num_devices):
                if num_times_comm_matrix[nccl_type][dst][src] != 0:
                    ret[nccl_type].append([src, dst, num_bytes_comm_matrix[nccl_type][dst][src]])

    return ret

def print_ret(ret):
    for nccl_type in ret.keys():
        print("--------------------------------------------")
        print("NCCL {} Communitation : ".format(nccl_type))
        for data in ret[nccl_type]:
            print("From GPU {} to GPU {}, Message Size {} GB.".format(
                data[0], data[1], data[2] / (1024 ** 3)
            ))

def plot_cdf():
    pdf = final / np.sum(final)
    cdf=np.cumsum(pdf)

    plt.plot(num, pdf, marker="o",label="PMF")
    plt.plot(num,cdf,marker="o",label="CDF")
    plt.xlim(0, num[-1])
    plt.ylim(0, 1.1)
    plt.xlabel("Thoughput")
    plt.ylabel("Probability")
    plt.title("CDF for " + name)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(name, dpi=1601)

def get_id2device_map(CUDA_VISIBLE_DEVICES):
    tmp = CUDA_VISIBLE_DEVICES.split(',')
    for i in range(len(tmp)):
        tmp[i] = int(tmp[i])

    return tmp

if __name__ == '__main__':
    args = parse_args()

    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    
    id2device = get_id2device_map(CUDA_VISIBLE_DEVICES)

    NUM_GPU = 8
    ret = reduce(NUM_GPU, id2device, filepath_prefix=args.directory + "comscribe_*_*.csv")
    print_ret(ret)
    cdf = allreduce_cdf(NUM_GPU, id2device, filepath_prefix=args.directory + "comscribe_reducescatter_*.csv")

    pprint.pprint("Allreduce CDF(use Allscatter to compute)", cdf)

    # with open(args.target, "w") as outfile:
    #     json.dump(ret, outfile, indent=4)