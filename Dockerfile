# CUDA base remains configurable via CUDA_TAG
ARG CUDA_TAG
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-13 \
    libcudnn9-dev-cuda-13

RUN apt-get update && apt-get install -y \
    git ffmpeg python3.11 python3.11-venv python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Modern PyTorch: install from the cu128 wheel index
# Pin to a recent trio known to ship cu128 wheels
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu13 torch torchvision torchaudio --upgrade


# App setup
WORKDIR /app
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ./src /app

# Useful envs
ENV PATH="/root/.local/bin:${PATH}"
ENV HF_HOME="/root/.cache/huggingface"

# App entry
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
