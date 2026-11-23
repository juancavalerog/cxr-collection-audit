# Imagen del proyecto paper_new (CSD-Net): PyTorch con ruedas cu126, sin Jupyter y sin puertos.
# Driver del servidor: NVIDIA 560.35.03 (CUDA 12.6 como máximo). Las ruedas traen su propio CUDA y cuDNN.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 procps tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --retries 10 --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r /tmp/requirements.txt \
    && pip check

# Cachés de torch y Hugging Face (volúmenes con nombre) y HOME escribibles con cualquier UID
RUN mkdir -p /cache/torch /cache/hf /workspace/paper_new /data/covid \
    && chmod 1777 /cache /cache/torch /cache/hf
ENV TORCH_HOME=/cache/torch \
    HF_HOME=/cache/hf \
    HOME=/tmp

WORKDIR /workspace/paper_new
# tini es el PID 1: reenvía SIGTERM de docker stop al proceso de entrenamiento
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-c", "print('imagen paper-new:cu126; los trabajos se lanzan con runner/job_runner.py')"]
