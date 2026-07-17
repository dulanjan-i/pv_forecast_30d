# =============================================================================
# MiRACLE v1.0 — 30-Day PV Forecast Inference Container
# =============================================================================
# Build:  docker build -t miracle-inference:v1.0 .
# Run:    docker run miracle-inference:v1.0 /app/scripts/run_inference.sh --date 2026-01-02
# Shell:  docker run -it miracle-inference:v1.0 bash
# Verify: docker run --rm miracle-inference:v1.0 shasum -a 256 -c /app/checkpoints/CHECKPOINT_MANIFEST.sha256
# =============================================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements before code so pip install layer is cached independently
COPY requirements/requirements_docker.txt /app/requirements_docker.txt

# CPU-only torch — +cpu suffix only available from 2.6.0+ on ARM64 (aarch64)
RUN pip install --no-cache-dir \
        torch==2.7.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /app/requirements_docker.txt

COPY src/                                               /app/src/
COPY scripts/                                           /app/scripts/
COPY V1.0_FINAL_TFT/                                   /app/V1.0_FINAL_TFT/
COPY freeze/final_thesis_v1/rl/ddqn_minenv_v2/         /app/checkpoints/rl_v2/
COPY freeze/CHECKPOINT_MANIFEST.sha256                  /app/checkpoints/CHECKPOINT_MANIFEST.sha256

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MIRACLE_RL_CKPT=/app/checkpoints/rl_v2/ddqn_best.pt
ENV MIRACLE_SHORT_CKPT=/app/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt
ENV MIRACLE_LONG_CKPT=/app/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt
ENV MIRACLE_PLANT_META=/app/V1.0_FINAL_TFT/plant_metadata/plant_03.json

RUN chmod +x /app/scripts/run_inference.sh

HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD python -c "from src.inference.physics_aware_forecaster import PhysicsAwareForecaster; print('OK')" \
    || exit 1

CMD ["python", "-m", "src.inference.physics_aware_forecaster", "--help"]
