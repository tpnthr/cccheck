# Use a PyTorch CUDA 13.0 image
FROM pytorch/pytorch:2.8.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y git ffmpeg python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ./src /app

# Useful envs
ENV PATH="/root/.local/bin:${PATH}"
ENV HF_HOME="/root/.cache/huggingface"

# App entry
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
