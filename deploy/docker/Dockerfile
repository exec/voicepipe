# voicepipe worker — runs `voicepipe serve` on a public port with a per-boot
# random auth token. Suitable as a vast.ai / runpod / hetzner template, or any
# place you can `docker run`. Self-contained: bundles torch (cu128/Blackwell),
# the [train] / [deploy] extras, and a llama.cpp checkout for LoRA→GGUF.
#
# Build from the repo root:
#   ./deploy/docker/build-worker.sh
# or
#   docker build -f deploy/docker/Dockerfile -t ghcr.io/exec/voicepipe-worker:latest .
#
# Released images live at ghcr.io/exec/voicepipe-worker — built and pushed
# automatically by .github/workflows/release.yml on every `v*` tag.

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.hf_home

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/voicepipe

# Pip-installable surface only — corpora / datasets / scratch never enter the image.
COPY pyproject.toml constraints-train.txt ./
COPY pipeline ./pipeline

# torch must be installed from PyTorch's cu128 index (Blackwell sm_120 / 5090).
# For Ada/Ampere (3090/4090), substitute torch==2.5.1 from .../cu121 — the rest
# of the pins in constraints-train.txt are CUDA-agnostic.
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 \
 && /opt/venv/bin/pip install -e ".[gui,train,deploy]" -c constraints-train.txt \
 && /opt/venv/bin/pip install hf_transfer

# llama.cpp for the deploy stage's convert_lora_to_gguf.py.
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
 && /opt/venv/bin/pip install -r /opt/llama.cpp/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}" \
    VOICEPIPE_LLAMA_CPP_DIR=/opt/llama.cpp \
    VOICEPIPE_PORT=8765

# /workspace is the writable home for HF cache, projects, datasets, the pairing-block file.
WORKDIR /workspace

COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]
