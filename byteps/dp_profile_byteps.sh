# now you are in docker environment
# export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234

if [[ -z "${SUFFIX}" ]]; then
    echo "ERROR:"
    echo "  You must give me SUFFIX environment variable."
    exit -1
fi

python -c "import ptflops" > /dev/null 2> /dev/null
retVal=$?
if [ $retVal -ne 0 ]; then
    pip install ptflops
fi

export MODEL_NAME=resnet50_${SUFFIX}

mkdir -p ./nsys_output/
mkdir -p ./nccl_profiles/${MODEL_NAME}/

LD_PRELOAD=/usr/local/ComScribe/nccl/build/lib/libnccl.so \
    nsys profile --wait primary --force-overwrite true -o ./nsys_output/${MODEL_NAME} \
    bpslaunch python3 ./benchmark_byteps.py --model resnet50 --num-iters 10 ${PROFILE_ARGS}

if test -n "$(find ./ -maxdepth 1 -name 'comscribe_*_*.csv' -print -quit)"
then
    mv ./comscribe_*_*.csv ./nccl_profiles/${MODEL_NAME}/
    # python ./nccl_utils.py -d ./nccl_profiles/${MODEL_NAME}/ -t ./comscribe_output/${MODEL_NAME}.json
fi

nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column \
    --output .,- ./nsys_output/${MODEL_NAME}.nsys-rep