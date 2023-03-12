export BYTEPS_TRACE_ON=1
# export BYTEPS_TRACE_END_STEP=20
# export BYTEPS_TRACE_START_STEP=10
export BYTEPS_TRACE_DIR=./traces_2

# export PS_VERBOSE=2

# Already set in docker config.
# export NVIDIA_VISIBLE_DEVICES=2,3,6,7
# export CUDA_VISIBLE_DEVICES=2,3,6,7

export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.17.0.2 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

export MODEL_NAME=resnet50

mkdir -p ./nsys_output/
nsys profile --wait primary --force-overwrite true -o ./nsys_output/${MODEL_NAME}_${DMLC_WORKER_ID} \
    bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model ${MODEL_NAME} --num-iters 10

nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column \
    --output .,- ./nsys_output/${MODEL_NAME}_${DMLC_WORKER_ID}.nsys-rep
