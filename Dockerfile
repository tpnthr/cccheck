# Use NVIDIA CUDA 13.0 base image
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install PyTorch nightly for CUDA 13
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu13

# App setup
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source code
COPY ./src /app

# Useful envs
ENV HF_HOME="/root/.cache/huggingface"

# App entry
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
