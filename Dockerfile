ARG CUDA_TAG=2.7.0
ARG CUDA_VERSION=12.8

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime


WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt ./
COPY ./src /app

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y torchaudio && pip install torchaudio==2.0.2+cu128 --index-url https://download.pytorch.org/whl/cu128

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
