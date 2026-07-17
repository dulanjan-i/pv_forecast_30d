# Docker Guide — MiRACLE Inference Container

This document explains the Dockerfile decisions in detail.
Useful for onboarding, interviews, or understanding the containerisation choices.

---

## Quick Reference

```bash
# Build
docker build -t miracle-inference:v1.0 .

# Run inference (live weather from OpenMeteo)
docker run -v $(pwd)/outputs:/app/outputs miracle-inference:v1.0 \
  /app/scripts/run_inference.sh --date 2026-01-02

# Interactive shell inside the container
docker run -it miracle-inference:v1.0 bash

# Verify checkpoint integrity inside the container
docker run --rm miracle-inference:v1.0 \
  shasum -a 256 -c /app/checkpoints/CHECKPOINT_MANIFEST.sha256
```

---

## Dockerfile Decisions Explained

### Base image — `python:3.11-slim`
- `slim` = minimal Debian (~150MB) vs full Python image (~900MB)
- Python 3.11 matches the conda environment used during development
- `pytorch-forecasting` 1.x is tested against 3.11

### System packages
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*
```
- `libgomp1` — OpenMP, required by scikit-learn CPU kernels
- `build-essential` — C compiler for Python extension builds (pvlib, scipy)
- Cleanup (`rm -rf /var/lib/apt/lists/*`) **must** be in the same `RUN` layer
  or the apt cache bloats the image permanently (Docker layers are immutable)

### Layer cache ordering — requirements before code
```dockerfile
COPY requirements/requirements_docker.txt .   # ← copy first
RUN pip install ...                            # ← install
COPY src/ ...                                  # ← copy code last
```
Docker caches each layer. If you copy all code first, any `.py` change
invalidates the pip install cache (slow ~5min rebuild every time).
Copying requirements first means pip only re-runs when `requirements_docker.txt`
itself changes — code edits skip straight to the COPY step (~seconds).

### CPU-only PyTorch
```dockerfile
RUN pip install torch==2.7.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```
- CPU keeps the image portable — no CUDA driver required on the host
- On CPU, 30-day inference takes ~30s (acceptable for demo / interview use)
- For GPU deployment: replace with `torch==2.7.1+cu121` from the CUDA index
- **ARM64 note:** the `+cpu` suffix only exists from PyTorch 2.6.0+ on aarch64.
  Earlier versions (e.g. 2.2.x) are available without the suffix but resolving
  them via the PyTorch index on ARM64 fails — use 2.6.0+ with explicit `+cpu`.
- `torchvision` excluded: not imported anywhere in the TFT/RL inference pipeline

### Environment variables
```dockerfile
ENV PYTHONPATH=/app               # lets Python find src/ without install
ENV PYTHONUNBUFFERED=1            # real-time log flushing in containers
ENV MIRACLE_RL_CKPT=...           # checkpoint paths configurable at runtime
```
All `MIRACLE_*` env vars can be overridden at `docker run` time:
```bash
docker run -e MIRACLE_RL_CKPT=/mnt/weights/custom.pt miracle-inference:v1.0
```

### Healthcheck
```dockerfile
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD python -c "from src.inference.physics_aware_forecaster import PhysicsAwareForecaster; print('OK')"
```
Docker (and Kubernetes) polls this periodically. If it fails, the container
is marked `unhealthy` and orchestrators can restart it automatically.
Verifies the full import chain (torch → lightning → pytorch-forecasting → src).

### CMD vs ENTRYPOINT
- `CMD` is used (not `ENTRYPOINT`) so the default can be easily overridden:
  ```bash
  docker run miracle-inference:v1.0 bash          # override CMD with bash
  docker run miracle-inference:v1.0 python script.py   # run any script
  ```
- `ENTRYPOINT` would force the binary and can only be overridden with `--entrypoint`

---

## Image contents

```
/app/
├── src/                          ← MiRACLE pipeline (inference, rl, models, ...)
├── scripts/run_inference.sh      ← CLI wrapper with SHA256 integrity check
├── V1.0_FINAL_TFT/
│   ├── shorthead_seed42/checkpoints/best.pt   ← TFT short-head (1.73MB)
│   ├── longhead_seed43/checkpoints/best.pt    ← TFT long-head  (1.73MB)
│   ├── shorthead_seed42/{column_roles,run_config}.json
│   ├── longhead_seed43/{column_roles,run_config}.json
│   └── plant_metadata/plant_03.json
└── checkpoints/
    ├── rl_v2/ddqn_best.pt        ← RL DDQN v2 meta-controller (110KB)
    └── CHECKPOINT_MANIFEST.sha256
```

---

## Deploying for GPU inference

Replace the torch install step in the Dockerfile:

```dockerfile
# GPU — CUDA 12.1
RUN pip install --no-cache-dir \
        torch==2.7.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

And use the NVIDIA base image for production:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

Then pass `--device` flag to use the GPU:
```bash
docker run --gpus all miracle-inference:v1.0-gpu \
  /app/scripts/run_inference.sh --date 2026-01-02 --device cuda
```

---

## Platform notes (ARM64 vs AMD64)

Built on Apple Silicon (aarch64), the image is `linux/arm64`.
For deployment on x86 servers (AWS, Azure, GCP), build with:

```bash
docker buildx build --platform linux/amd64 -t miracle-inference:v1.0-amd64 .
```

Or use GitHub Actions (Step 2 of the roadmap) which builds for both platforms
via `docker/build-push-action` with `platforms: linux/amd64,linux/arm64`.
