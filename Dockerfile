# =============================================================================
# MiRACLE v1.0 Inference Container
# =============================================================================
# WHAT IS A DOCKERFILE?
# A Dockerfile is a recipe. Each line is an instruction that creates a "layer"
# in the final image. Docker caches layers — if a layer hasn't changed,
# it reuses it, making rebuilds fast.
#
# HOW TO BUILD:
#   docker build -t miracle-inference:v1.0 .
#
# HOW TO RUN:
#   # Interactive shell inside the container:
#   docker run -it miracle-inference:v1.0 bash
#
#   # Run a 30-day forecast (live weather from OpenMeteo):
#   docker run miracle-inference:v1.0 \
#     python -m src.inference.physics_aware_forecaster \
#     --forecast-start "2026-01-02" \
#     --use-live-weather \
#     --output-file /tmp/forecast.parquet
#
#   # Run via the helper script:
#   docker run -v $(pwd)/outputs:/app/outputs miracle-inference:v1.0 \
#     /app/scripts/run_inference.sh --date 2026-01-02
# =============================================================================


# ── STAGE 1: Base image ───────────────────────────────────────────────────────
# FROM tells Docker which base image to start from.
# We use python:3.11-slim — "slim" = minimal Debian with Python pre-installed.
# This is ~150MB instead of ~900MB for the full Python image.
# Why 3.11? That's what the conda env uses; pytorch-forecasting 1.4.0 is
# tested on 3.11.
FROM python:3.11-slim

# ── STAGE 2: System-level dependencies ───────────────────────────────────────
# RUN executes a shell command inside the container during BUILD time.
# We chain commands with && and clean up in the same layer (crucial for
# keeping image size small — each RUN creates a layer, so cleanup must be
# in the same RUN or it doesn't help).
#
# What we need:
# - git: some pip packages install from git at build time
# - libgomp1: OpenMP, required by LightGBM / XGBoost CPU kernels
# - curl: useful for healthchecks
# - build-essential: C compiler, needed for some Python extension builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
    # ^ IMPORTANT: always delete apt cache in the same RUN layer or the
    #   cache files bloat the image permanently.


# ── STAGE 3: Set the working directory ───────────────────────────────────────
# WORKDIR sets the default directory for all subsequent commands.
# Like doing `cd /app` permanently. /app is a convention for containerised apps.
WORKDIR /app


# ── STAGE 4: Install Python dependencies ─────────────────────────────────────
# WHY COPY requirements first, then COPY the rest of the code?
# Docker caches each layer. If you copy ALL your code first, then install
# deps, any code change invalidates the pip install layer (slow rebuild).
# By copying requirements.txt first, pip install is only re-run when
# requirements.txt itself changes — not when you edit a .py file.
# This is the most important Docker optimisation trick.
COPY requirements/requirements_docker.txt /app/requirements_docker.txt

# Install CPU-only PyTorch first (from the official PyTorch index).
# We separate this from the rest because it's large and has a special index URL.
# --index-url tells pip to look at the PyTorch CDN instead of PyPI.
RUN pip install --no-cache-dir \
        torch==2.7.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
    # torchvision excluded: not used by TFT/RL inference pipeline.
    # ARM64 +cpu suffix available from 2.6.0+ only.

# Install the rest of the inference dependencies.
# --no-cache-dir: don't cache downloaded wheels inside the container (saves space).
RUN pip install --no-cache-dir -r /app/requirements_docker.txt


# ── STAGE 5: Copy the application code ───────────────────────────────────────
# COPY <source-on-host> <destination-in-container>
# The .dockerignore file controls what gets excluded.
# We copy src/ (the Python pipeline), scripts/ (CLI wrappers),
# V1.0_FINAL_TFT/ (TFT model configs + weights), and the RL checkpoint.
COPY src/                          /app/src/
COPY scripts/                      /app/scripts/
COPY V1.0_FINAL_TFT/               /app/V1.0_FINAL_TFT/
COPY freeze/final_thesis_v1/rl/ddqn_minenv_v2/  /app/checkpoints/rl_v2/
COPY freeze/CHECKPOINT_MANIFEST.sha256           /app/checkpoints/CHECKPOINT_MANIFEST.sha256


# ── STAGE 6: Environment variables ───────────────────────────────────────────
# ENV sets environment variables that persist into the running container.
# PYTHONPATH tells Python where to find our src/ modules (like sys.path).
# PYTHONUNBUFFERED=1 forces stdout/stderr to flush immediately — important
# for seeing logs in real time when running in Docker.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MIRACLE_RL_CKPT=/app/checkpoints/rl_v2/ddqn_best.pt
ENV MIRACLE_SHORT_CKPT=/app/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt
ENV MIRACLE_LONG_CKPT=/app/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt
ENV MIRACLE_PLANT_META=/app/V1.0_FINAL_TFT/plant_metadata/plant_03.json


# ── STAGE 7: Make the helper script executable ───────────────────────────────
# Files copied from Mac often lose their executable bit.
# chmod +x restores it so `docker run miracle-inference` can run the script.
RUN chmod +x /app/scripts/run_inference.sh


# ── STAGE 8: Healthcheck ─────────────────────────────────────────────────────
# HEALTHCHECK tells Docker how to test if the container is working.
# Docker will run this command periodically; if it fails, the container
# is marked "unhealthy". Useful when running in Kubernetes or Docker Compose.
# Here we just verify the pipeline imports cleanly.
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD python -c "from src.inference.physics_aware_forecaster import PhysicsAwareForecaster; print('OK')" \
    || exit 1


# ── STAGE 9: Default command ──────────────────────────────────────────────────
# CMD is what runs when you do `docker run miracle-inference` with no extra args.
# ENTRYPOINT vs CMD:
#   ENTRYPOINT = always runs (can't be overridden without --entrypoint flag)
#   CMD        = default args, easily overridden: `docker run image other_cmd`
# We use CMD so the user can override with `docker run miracle-inference bash`
# for interactive debugging.
CMD ["python", "-m", "src.inference.physics_aware_forecaster", "--help"]
