#!/bin/bash

# do a snapshot of experiment
EXP_NAME=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p ./results/${EXP_NAME}/
BASE_FOLDER=pytorch

python ./reduce_data.py --csv=./${BASE_FOLDER}/nsys_output/resnet50_dp_1_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_dp_1.png

python ./reduce_data.py --csv=./${BASE_FOLDER}/nsys_output/resnet50_dp_2_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_dp_2.png

python ./reduce_data.py --csv=./${BASE_FOLDER}/nsys_output/resnet50_dp_4_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_dp_4.png

python ./reduce_data.py --csv=./${BASE_FOLDER}/nsys_output/resnet50_dp_8_gputrace.csv --fig_name=./results/${EXP_NAME}/resnet50_dp_8.png

