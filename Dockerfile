# System
FROM ghcr.io/nvidia/driver:56b85890-550.90.07-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y \
  git vim curl software-properties-common \
  libglew-dev wget \
  && apt-get clean

# Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools

# Envs
RUN pip install dm_control
ENV MUJOCO_GL=egl

# Requirements
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install elements hydra-core tensordict pandas tqdm gym torchrl
RUN pip install termcolor
ENV MUJOCO_EGL_DEVICE_ID=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    && apt-get clean

ENV CUDA_VISIBLE_DEVICES=0

# Source
RUN mkdir /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root .

# Cloud
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600

WORKDIR /app/tdmpc2
ENTRYPOINT []
