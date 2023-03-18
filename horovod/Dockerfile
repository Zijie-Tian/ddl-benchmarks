FROM horovod/horovod:latest

ARG HOROVOD_BASE_PATH=/usr/local
ARG COMSCRIBE_PATH=$HOROVOD_BASE_PATH/ComScribe

RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2023.1.1 &&\
     rm -rf /var/lib/apt/lists/*

# Install Comscribe
RUN cd $HOROVOD_BASE_PATH && \
    git clone https://github.com/ParCoreLab/ComScribe.git &&\
    cd $COMSCRIBE_PATH &&\
    ./install.sh

