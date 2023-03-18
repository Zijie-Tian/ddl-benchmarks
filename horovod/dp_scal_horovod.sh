## Not use this var, because its sub script will use it
# export MODEL_NAME=resnet50

# 1 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/horovod:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0"' \
    -e SUFFIX='dp_1' \
    -e PROFILE_ARGS='--flops' \
    -e NGPU=1 \
    horovod/horovod_nsys bash -c "cd /workspace && /workspace/dp_profile_horovod.sh"

# # 2 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/horovod:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1"' \
    -e SUFFIX='dp_2' \
    -e PROFILE_ARGS='--flops' \
    -e NGPU=2 \
    horovod/horovod_nsys bash -c "cd /workspace && /workspace/dp_profile_horovod.sh"

# # 4 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/horovod:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3"' \
    -e SUFFIX='dp_4' \
    -e PROFILE_ARGS='--flops' \
    -e NGPU=4 \
    horovod/horovod_nsys bash -c "cd /workspace && /workspace/dp_profile_horovod.sh"

# # 8 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/horovod:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3,4,5,6,7"' \
    -e SUFFIX='dp_8' \
    -e PROFILE_ARGS='--flops' \
    -e NGPU=8 \
    horovod/horovod_nsys bash -c "cd /workspace && /workspace/dp_profile_horovod.sh"


