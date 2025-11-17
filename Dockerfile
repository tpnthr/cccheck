# Dockerfile

ARG CUDA_TAG=2.7.0
ARG CUDA_VERSION=12.8

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    pkg-config \
    # ↓↓↓ ДОБАВЛЯЕМ ПАКЕТЫ РАЗРАБОТЧИКА FFmpeg ↓↓↓
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY ./src /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]