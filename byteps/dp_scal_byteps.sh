export MODEL_NAME=resnet50

# 1 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/byteps:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0"' \
    -e SUFFIX='dp_1' \
    -e PROFILE_ARGS='--flops' \
    bytepsimage/pytorch_nsys bash -c "/workspace/dp_profile_byteps.sh"

# 2 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/byteps:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1"' \
    -e SUFFIX='dp_2' \
    -e PROFILE_ARGS='--flops' \
    bytepsimage/pytorch_nsys bash -c "/workspace/dp_profile_byteps.sh"

# 4 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/byteps:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3"' \
    -e SUFFIX='dp_4' \
    -e PROFILE_ARGS='--flops' \
    bytepsimage/pytorch_nsys bash -c "/workspace/dp_profile_byteps.sh"

# 8 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/byteps:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3,4,5,6,7"' \
    -e SUFFIX='dp_8' \
    -e PROFILE_ARGS='--flops' \
    bytepsimage/pytorch_nsys bash -c "/workspace/dp_profile_byteps.sh"


