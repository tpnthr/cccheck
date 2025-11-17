ARG CUDA_TAG=2.7.0
ARG CUDA_VERSION=12.8

FROM pytorch/pytorch:${CUDA_TAG}-cuda${CUDA_VERSION}-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt ./
COPY ./src /app

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade torchaudio

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
