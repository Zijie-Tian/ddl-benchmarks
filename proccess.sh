#!/bin/bash

# do a snapshot of experiment
EXP_NAME=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p ./results/${EXP_NAME}/

python ./reduce_data.py --csv=./byteps/nsys_output/resnet50_0_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_0.png

python ./reduce_data.py --csv=./byteps/nsys_output/resnet50_1_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_1.png

