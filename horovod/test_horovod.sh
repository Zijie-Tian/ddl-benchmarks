export MODEL_NAME=resnet50_dp

mkdir -p ./nsys_output/
mkdir -p ./nccl_profiles/${MODEL_NAME}/

rm ./nsys_output/${MODEL_NAME}*
rm ./nccl_profiles/${MODEL_NAME}/*

python -c "import ptflops" > /dev/null 2> /dev/null
retVal=$?
if [ $retVal -ne 0 ]; then
    pip install ptflops
fi

LD_PRELOAD=/usr/local/ComScribe/nccl/build/lib/libnccl.so \
    nsys profile --wait primary --force-overwrite true -o ./nsys_output/${MODEL_NAME} \
    horovodrun -np 4 -H localhost:4 python3 ./imagenet_benchmark.py --model resnet50 --num-iters 10

if test -n "$(find ./ -maxdepth 1 -name 'comscribe_*_*.csv' -print -quit)"
then
    mv ./comscribe_*_*.csv ./nccl_profiles/${MODEL_NAME}/
    # python ./nccl_utils.py -d ./nccl_profiles/${MODEL_NAME}/ -t ./comscribe_output/${MODEL_NAME}.json
fi

nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column \
    --output .,- ./nsys_output/${MODEL_NAME}.nsys-rep