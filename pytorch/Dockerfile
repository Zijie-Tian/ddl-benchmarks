FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG BASE_PATH=/usr/local
ARG COMSCRIBE_PATH=$BASE_PATH/ComScribe

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        numactl \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        net-tools \
        cmake \
        lsb-release \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools

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
RUN cd $BASE_PATH && \
    git clone https://github.com/ParCoreLab/ComScribe.git &&\
    cd $COMSCRIBE_PATH &&\
    ./install.sh

