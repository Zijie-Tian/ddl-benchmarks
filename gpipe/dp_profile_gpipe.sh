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

python -c "import torchgpipe" > /dev/null 2> /dev/null
retVal=$?
if [ $retVal -ne 0 ]; then
    pip install torchgpipe
fi

export MODEL_NAME=resnet50_${SUFFIX}

## In the docker
mkdir -p ./nsys_output/
mkdir -p ./nccl_profiles/${MODEL_NAME}/
mkdir -p ./logs

rm -f ./logs/${MODEL_NAME}.log
rm ./nsys_output/${MODEL_NAME}*
rm -f ./nccl_profiles/${MODEL_NAME}/* || true

LD_PRELOAD=/usr/local/ComScribe/nccl/build/lib/libnccl.so \
    nsys profile --wait primary --force-overwrite true -o ./nsys_output/${MODEL_NAME} \
    python3 ./imagenet_benchmark.py --model resnet50 --num-batches-per-iter 10 --num-iter 10 \
    --batch-size ${BATCH_SIZE} --partitions ${PARTITIONS} --chunks ${CHUNKS} ${PROFILE_ARGS} \
    1> ./logs/${MODEL_NAME}.log 2> ./logs/${MODEL_NAME}_ERROR.log

if test -n "$(find ./ -maxdepth 1 -name 'comscribe_*_*.csv' -print -quit)"
then
    mv ./comscribe_*_*.csv ./nccl_profiles/${MODEL_NAME}/
    # python ./nccl_utils.py -d ./nccl_profiles/${MODEL_NAME}/ -t ./comscribe_output/${MODEL_NAME}.json
fi

nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column \
    --output .,- ./nsys_output/${MODEL_NAME}.nsys-rep