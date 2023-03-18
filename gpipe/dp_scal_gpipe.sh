export MODEL_NAME=resnet50

# 1 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/gpipe:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0"' \
    -e SUFFIX='pp_1' \
    -e BATCH_SIZE='128' \
    -e PARTITIONS='1' \
    -e CHUNKS='16' \
    -e PROFILE_ARGS='--flops' \
    tzj/pytorch_nsys bash -c "/workspace/dp_profile_gpipe.sh"

# # 2 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/gpipe:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1"' \
    -e SUFFIX='pp_2' \
    -e BATCH_SIZE='128' \
    -e PARTITIONS='2' \
    -e CHUNKS='16' \
    -e PROFILE_ARGS='--flops' \
    tzj/pytorch_nsys bash -c "/workspace/dp_profile_gpipe.sh"

# # 4 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/gpipe:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3"' \
    -e SUFFIX='pp_4' \
    -e BATCH_SIZE='128' \
    -e PARTITIONS='4' \
    -e CHUNKS='16' \
    -e PROFILE_ARGS='--flops' \
    tzj/pytorch_nsys bash -c "/workspace/dp_profile_gpipe.sh"

# # 8 worker
nvidia-docker run -it --rm -v=/home/tzj/Code/ddl-benchmarks/gpipe:/workspace \
    --shm-size=327680m --cap-add SYS_ADMIN --gpus '"device=0,1,2,3,4,5,6,7"' \
    -e SUFFIX='pp_8' \
    -e BATCH_SIZE='128' \
    -e PARTITIONS='8' \
    -e CHUNKS='16' \
    -e PROFILE_ARGS='--flops' \
    tzj/pytorch_nsys bash -c "/workspace/dp_profile_gpipe.sh"


