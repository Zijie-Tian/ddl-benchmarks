import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(csv_file, name = ""):
    if csv_file == None or csv_file[-4:] != ".csv":
        print("Not a valid file name")
        exit(-1)

    data_read = pd.read_csv(csv_file)
    data_read = data_read[data_read['Name'].isin(['[CUDA memcpy HtoD]', '[CUDA memcpy DtoH]'])][['Duration (ns)', 'Bytes (MB)', 'Throughput (MBps)']]
    # data_read = data_read[data_read['Name'].isin(['[CUDA memcpy HtoD]'])][['Duration (ns)', 'Bytes (MB)', 'Throughput (MBps)']]
    data_read['Throughput (MBps)'] = data_read['Throughput (MBps)'] / 1024

    i = 0
    bins = []
    num = []
    while i <= max(data_read['Throughput (MBps)']) + 0.2:
        num.append(i+0.1)
        bins.append(i)
        i += 0.2
    num.pop()
        
    segments = pd.cut(data_read['Throughput (MBps)'], bins, right=False)
    counts=list(pd.value_counts(segments, sort=False))

    final = [0] * len(num)
    for i in range(data_read.shape[0]):
        for j in range(len(bins) - 1):
            if data_read['Throughput (MBps)'].iloc[i] >= bins[j] and data_read['Throughput (MBps)'].iloc[i] <= bins[j+1]:
                final[j] += float(data_read['Bytes (MB)'].iloc[i])
                break

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

def parse_args():
    # HIDDEN_DIM, NUM_LAYER, NUM_HEAD, BATCH_SIZE
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv', type=str,
                        help='GPU trace csv file name.')
    parser.add_argument('--fig_name', type=str, default="", help='result file name.')
    parser.add_argument('-d', '--dir', type=str, default="./")
    parser.add_argument('-r', '--reduce', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.csv, name=args.fig_name)