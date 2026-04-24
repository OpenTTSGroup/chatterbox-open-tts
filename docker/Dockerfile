# syntax=docker/dockerfile:1.7

# ============================================================================
# Build arguments
#   CUDA_VERSION    — base image CUDA version. Chatterbox pins torch==2.6.0;
#                     cu124 wheels match 12.4.x bases cleanly.
#   CUDA_FLAVOR     — cuDNN tag segment. 12.4+ uses ``cudnn-runtime``.
#   UBUNTU_VERSION  — base OS.
#   PYTHON_VERSION  — Chatterbox README tests on 3.11; pyproject.toml says >=3.10.
#   TORCH_VERSION   — upstream Chatterbox requirement.
#   TORCH_CUDA_TAG  — PyTorch wheel index tag matching CUDA_VERSION.
# ============================================================================
ARG CUDA_VERSION=12.4.1
ARG CUDA_FLAVOR=cudnn-runtime
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.6.0
ARG TORCH_CUDA_TAG=cu124

# ============================================================================
# Builder: python venv with torch, Chatterbox engine, and the API layer.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCH_CUDA_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        build-essential \
        git \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

# PyTorch — single layer, matched to Chatterbox's torch==2.6.0 constraint.
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

# API layer dependencies (pydantic/fastapi/soundfile/av/pyyaml).
COPY docker/requirements.api.txt /tmp/requirements.api.txt
RUN pip install -r /tmp/requirements.api.txt

# Chatterbox engine — has a proper pyproject.toml + setuptools backend, so we
# install it directly. Its dependency list pulls in transformers, librosa,
# s3tokenizer, diffusers, conformer, resemble-perth (from git), pykakasi,
# spacy-pkuseg, omegaconf, pyloudnorm, gradio (dev/demo).
COPY engine /tmp/engine
RUN pip install /tmp/engine && rm -rf /tmp/engine

# Strip bytecode caches.
RUN find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Runtime: minimal image carrying only the venv + app/.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        ffmpeg \
        libsndfile1 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/api

COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
