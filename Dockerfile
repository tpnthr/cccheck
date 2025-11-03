# Use ARGs so we can substitute from build args
ARG CUDA_TAG=2.7.0
ARG CUDA_VERSION=12.8

FROM pytorch/pytorch:${CUDA_TAG}-cuda${CUDA_VERSION}-cudnn9-runtime

WORKDIR /app

COPY ./src /app

RUN pip install --no-cache-dir uvicorn fastapi torch torchvision torchaudio \
    && python -m pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]