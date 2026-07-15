***REMOVED*** =============================================================================
***REMOVED*** MiRACLE v1.0 Inference Container
***REMOVED*** =============================================================================
***REMOVED*** WHAT IS A DOCKERFILE?
***REMOVED*** A Dockerfile is a recipe. Each line is an instruction that creates a "layer"
***REMOVED*** in the final image. Docker caches layers — if a layer hasn't changed,
***REMOVED*** it reuses it, making rebuilds fast.
***REMOVED***
***REMOVED*** HOW TO BUILD:
***REMOVED***   docker build -t miracle-inference:v1.0 .
***REMOVED***
***REMOVED*** HOW TO RUN:
***REMOVED***   ***REMOVED*** Interactive shell inside the container:
***REMOVED***   docker run -it miracle-inference:v1.0 bash
***REMOVED***
***REMOVED***   ***REMOVED*** Run a 30-day forecast (live weather from OpenMeteo):
***REMOVED***   docker run miracle-inference:v1.0 \
***REMOVED***     python -m src.inference.physics_aware_forecaster \
***REMOVED***     --forecast-start "2026-01-02" \
***REMOVED***     --use-live-weather \
***REMOVED***     --output-file /tmp/forecast.parquet
***REMOVED***
***REMOVED***   ***REMOVED*** Run via the helper script:
***REMOVED***   docker run -v $(pwd)/outputs:/app/outputs miracle-inference:v1.0 \
***REMOVED***     /app/scripts/run_inference.sh --date 2026-01-02
***REMOVED*** =============================================================================


***REMOVED*** ── STAGE 1: Base image ───────────────────────────────────────────────────────
***REMOVED*** FROM tells Docker which base image to start from.
***REMOVED*** We use python:3.11-slim — "slim" = minimal Debian with Python pre-installed.
***REMOVED*** This is ~150MB instead of ~900MB for the full Python image.
***REMOVED*** Why 3.11? That's what the conda env uses; pytorch-forecasting 1.4.0 is
***REMOVED*** tested on 3.11.
FROM python:3.11-slim

***REMOVED*** ── STAGE 2: System-level dependencies ───────────────────────────────────────
***REMOVED*** RUN executes a shell command inside the container during BUILD time.
***REMOVED*** We chain commands with && and clean up in the same layer (crucial for
***REMOVED*** keeping image size small — each RUN creates a layer, so cleanup must be
***REMOVED*** in the same RUN or it doesn't help).
***REMOVED***
***REMOVED*** What we need:
***REMOVED*** - git: some pip packages install from git at build time
***REMOVED*** - libgomp1: OpenMP, required by LightGBM / XGBoost CPU kernels
***REMOVED*** - curl: useful for healthchecks
***REMOVED*** - build-essential: C compiler, needed for some Python extension builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
    ***REMOVED*** ^ IMPORTANT: always delete apt cache in the same RUN layer or the
    ***REMOVED***   cache files bloat the image permanently.


***REMOVED*** ── STAGE 3: Set the working directory ───────────────────────────────────────
***REMOVED*** WORKDIR sets the default directory for all subsequent commands.
***REMOVED*** Like doing `cd /app` permanently. /app is a convention for containerised apps.
WORKDIR /app


***REMOVED*** ── STAGE 4: Install Python dependencies ─────────────────────────────────────
***REMOVED*** WHY COPY requirements first, then COPY the rest of the code?
***REMOVED*** Docker caches each layer. If you copy ALL your code first, then install
***REMOVED*** deps, any code change invalidates the pip install layer (slow rebuild).
***REMOVED*** By copying requirements.txt first, pip install is only re-run when
***REMOVED*** requirements.txt itself changes — not when you edit a .py file.
***REMOVED*** This is the most important Docker optimisation trick.
COPY requirements/requirements_docker.txt /app/requirements_docker.txt

***REMOVED*** Install CPU-only PyTorch first (from the official PyTorch index).
***REMOVED*** We separate this from the rest because it's large and has a special index URL.
***REMOVED*** --index-url tells pip to look at the PyTorch CDN instead of PyPI.
RUN pip install --no-cache-dir \
        torch==2.7.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
    ***REMOVED*** torchvision excluded: not used by TFT/RL inference pipeline.
    ***REMOVED*** ARM64 +cpu suffix available from 2.6.0+ only.

***REMOVED*** Install the rest of the inference dependencies.
***REMOVED*** --no-cache-dir: don't cache downloaded wheels inside the container (saves space).
RUN pip install --no-cache-dir -r /app/requirements_docker.txt


***REMOVED*** ── STAGE 5: Copy the application code ───────────────────────────────────────
***REMOVED*** COPY <source-on-host> <destination-in-container>
***REMOVED*** The .dockerignore file controls what gets excluded.
***REMOVED*** We copy src/ (the Python pipeline), scripts/ (CLI wrappers),
***REMOVED*** V1.0_FINAL_TFT/ (TFT model configs + weights), and the RL checkpoint.
COPY src/                          /app/src/
COPY scripts/                      /app/scripts/
COPY V1.0_FINAL_TFT/               /app/V1.0_FINAL_TFT/
COPY freeze/final_thesis_v1/rl/ddqn_minenv_v2/  /app/checkpoints/rl_v2/
COPY freeze/CHECKPOINT_MANIFEST.sha256           /app/checkpoints/CHECKPOINT_MANIFEST.sha256


***REMOVED*** ── STAGE 6: Environment variables ───────────────────────────────────────────
***REMOVED*** ENV sets environment variables that persist into the running container.
***REMOVED*** PYTHONPATH tells Python where to find our src/ modules (like sys.path).
***REMOVED*** PYTHONUNBUFFERED=1 forces stdout/stderr to flush immediately — important
***REMOVED*** for seeing logs in real time when running in Docker.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MIRACLE_RL_CKPT=/app/checkpoints/rl_v2/ddqn_best.pt
ENV MIRACLE_SHORT_CKPT=/app/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt
ENV MIRACLE_LONG_CKPT=/app/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt
ENV MIRACLE_PLANT_META=/app/V1.0_FINAL_TFT/plant_metadata/plant_03.json


***REMOVED*** ── STAGE 7: Make the helper script executable ───────────────────────────────
***REMOVED*** Files copied from Mac often lose their executable bit.
***REMOVED*** chmod +x restores it so `docker run miracle-inference` can run the script.
RUN chmod +x /app/scripts/run_inference.sh


***REMOVED*** ── STAGE 8: Healthcheck ─────────────────────────────────────────────────────
***REMOVED*** HEALTHCHECK tells Docker how to test if the container is working.
***REMOVED*** Docker will run this command periodically; if it fails, the container
***REMOVED*** is marked "unhealthy". Useful when running in Kubernetes or Docker Compose.
***REMOVED*** Here we just verify the pipeline imports cleanly.
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD python -c "from src.inference.physics_aware_forecaster import PhysicsAwareForecaster; print('OK')" \
    || exit 1


***REMOVED*** ── STAGE 9: Default command ──────────────────────────────────────────────────
***REMOVED*** CMD is what runs when you do `docker run miracle-inference` with no extra args.
***REMOVED*** ENTRYPOINT vs CMD:
***REMOVED***   ENTRYPOINT = always runs (can't be overridden without --entrypoint flag)
***REMOVED***   CMD        = default args, easily overridden: `docker run image other_cmd`
***REMOVED*** We use CMD so the user can override with `docker run miracle-inference bash`
***REMOVED*** for interactive debugging.
CMD ["python", "-m", "src.inference.physics_aware_forecaster", "--help"]
