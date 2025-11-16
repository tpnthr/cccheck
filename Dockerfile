# No need for CUDA_TAG here; we use a ready-made runtime with CUDA + cuDNN + PyTorch
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg python3.11 python3.11-venv python3-pip \
  && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app

# If you have requirements.txt, keep this block
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src /app

# Useful envs
ENV PATH="/root/.local/bin:${PATH}"
ENV HF_HOME="/root/.cache/huggingface"

# App entry
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
