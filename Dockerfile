# CUDA base remains configurable via CUDA_TAG
ARG CUDA_TAG
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-13 \
    libcudnn9-dev-cuda-13

RUN apt-get update && apt-get install -y \
    git ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Modern PyTorch: install from the cu128 wheel index
# Pin to a recent trio known to ship cu128 wheels
RUN apt-get install -y python3.9 python3.9-venv python3-pip
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu13


# App setup
WORKDIR /app
COPY requirements.txt ./
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ./src /app

# Useful envs
ENV PATH="/root/.local/bin:${PATH}"
ENV HF_HOME="/root/.cache/huggingface"

# App entry
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
