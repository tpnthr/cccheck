# Dockerfile
ARG CUDA_TAG
FROM nvidia/cuda:${CUDA_TAG}

ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg python3.11 python3.11-venv python3-pip \
  && rm -rf /var/lib/apt/lists/*

# PyTorch matching the CUDA version
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    if [ -z "${CUDA_VERSION}" ]; then echo "CUDA_VERSION is empty"; exit 1; fi && \
    if [ "${CUDA_VERSION}" = "12.4" ]; then \
      PYTORCH_CUDA_SUFFIX=cu124; \
    elif [ "${CUDA_VERSION}" = "12.1" ]; then \
      PYTORCH_CUDA_SUFFIX=cu121; \
    else \
      echo "Unsupported CUDA_VERSION=${CUDA_VERSION}. Update mapping."; exit 1; \
    fi && \
    python3.11 -m pip install --no-cache-dir \
      torch==2.1.2+${PYTORCH_CUDA_SUFFIX} \
      torchvision==0.16.2+${PYTORCH_CUDA_SUFFIX} \
      torchaudio==2.1.2+${PYTORCH_CUDA_SUFFIX} \
      -f https://download.pytorch.org/whl/torch_stable.html

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
