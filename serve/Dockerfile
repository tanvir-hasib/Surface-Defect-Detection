# PatchCore anomaly-detection API (TensorRT / ONNX Runtime) for an RTX 2080 Ti (Turing, sm_75)
# The host needs: NVIDIA driver R580+ and the NVIDIA Container Toolkit.
ARG BASE_IMAGE=nvidia/cuda:13.4.3-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv curl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

WORKDIR /srv
# Install dependencies first so code edits don't invalidate this (large) layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /cache /models && chown -R appuser /cache
USER appuser

ENV MODEL_DIR=/models \
    ENGINE_CACHE=/cache \
    BACKEND=trt \
    PRECISION=fp16 \
    CATEGORIES=all \
    MAX_BATCH=1

EXPOSE 8000
# First start builds TensorRT engines (~1 min per category), hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# One worker = one CUDA context holding all engines. Scale with more containers/GPUs, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
