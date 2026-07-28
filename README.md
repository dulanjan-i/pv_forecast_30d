# MiRACLE v1.0: Physics-Informed Hierarchical Learning for Long-Horizon Photovoltaic Power Forecasting

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20177800.svg)](https://doi.org/10.5281/zenodo.20177800)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker Build](https://github.com/dulanjan-i/pv_forecast_30d/actions/workflows/docker-build.yml/badge.svg)](https://github.com/dulanjan-i/pv_forecast_30d/actions/workflows/docker-build.yml)
[![CI — pytest](https://github.com/dulanjan-i/pv_forecast_30d/actions/workflows/ci.yml/badge.svg)](https://github.com/dulanjan-i/pv_forecast_30d/actions/workflows/ci.yml)

This repository contains the advanced **dual-head Temporal Fusion Transformer (TFT)** engine developed for my Master's thesis. It is made available for academic research, peer review, and technical evaluation.

![MiRACLE Architecture](architecture%20diagrams/core_architecture.html)

## Abstract
As solar photovoltaics (PV) supply an increasing share of electricity, grid operators depend on highly reliable forecasts. However, pure data-driven models often produce physically impossible outputs, such as non-zero power at night or values exceeding plant capacity.

This repository introduces **MiRACLE v1.0** (Meta-Intelligent Reinforcement-driven Adaptive Control for Learning-based Ensembles), a hybrid time-series forecasting approach that integrates deep learning with physics-informed constraints. Evaluated under a strict, forward-looking blind test for a commercial German PV plant, MiRACLE forcefully fuses short-horizon ramp accuracy with long-horizon shape fidelity to deliver physically plausible 30-day forecasts at a 15-minute resolution. Furthermore, this research explores the limitations of adaptive blending via Reinforcement Learning (RL), highlighting the critical dangers of metric exploitation where standard error metrics (like RMSE) can be artificially improved by sacrificing real-world operational value.

## Key Features
* **Dual-Head TFT Architecture**: Parallel independent instances structured to achieve cooperative fusion of outputs.
* **DDQN Meta-Controller**: An experimental offline Reinforcement Learning agent used to evaluate adaptive ensemble blending and expose the risks of metric exploitation in standard loss functions.
* **Physics Glue & Constraints**: A deterministic hierarchical inference layer that reconciles the TFT outputs using a PVLib-derived reference shape, enforcing operational validity through night-time masking and capacity clamping.
* **Robust Data Engineering**: Engineered to handle complex, messy, real-world data pipelines.

---

## Quick Start — Docker

The fastest way to run MiRACLE is via the pre-built image on GitHub Container Registry. No Python environment setup required.

**Requirements:** [Docker](https://docs.docker.com/get-docker/) installed (Desktop or Engine).

```bash
# Pull the latest image (supports linux/amd64 and linux/arm64)
docker pull ghcr.io/dulanjan-i/miracle-inference:latest

# Run a 30-day PV forecast (fetches live weather from OpenMeteo)
docker run -v $(pwd)/outputs:/app/outputs \
  ghcr.io/dulanjan-i/miracle-inference:latest \
  /app/scripts/run_inference.sh --date 2026-01-02

# Verify checkpoint integrity
docker run --rm ghcr.io/dulanjan-i/miracle-inference:latest \
  python -c "
import hashlib, os
for path in ['/app/checkpoints/rl_v2/ddqn_best.pt',
             '/app/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt',
             '/app/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt']:
    print('OK' if os.path.exists(path) else 'MISSING', os.path.basename(path))
"

# Interactive shell inside the container
docker run -it ghcr.io/dulanjan-i/miracle-inference:latest bash
```

> **Apple Silicon (M1/M2/M3):** The image includes a native `linux/arm64` build — no `--platform` flag needed.

> **GPU inference:** The image is CPU-only by default (~30s per 30-day forecast). See [docs/docker_guide.md](docs/docker_guide.md) for the GPU variant.

---

## Local Setup (Development)

```bash
git clone https://github.com/dulanjan-i/pv_forecast_30d.git
cd pv_forecast_30d

# Create conda environment
conda create -n pvforecast python=3.11
conda activate pvforecast
pip install -r requirements/requirements_calc02_frozen.txt

# Run tests
python -m pytest tests/test_rl_checkpoint.py -v
```

---

## Support & Documentation
* **Full Thesis**: [zenodo.org/records/20177801](https://zenodo.org/records/20177801) — complete methodology, literature review, and detailed results.
* **Docker Guide**: [docs/docker_guide.md](docs/docker_guide.md) — build options, GPU variant, ARM64 notes.
* **Data Schema**: [docs/final_data_cols.md](docs/final_data_cols.md)
* **Physics Glue**: [docs/glue.md](docs/glue.md)
* **Issues**: Open a GitHub issue or contact the author.

---

## Contributing
1. **Issue**: Open an issue describing the bug or feature request.
2. **Fork & Branch**: Create a feature branch from `main`.
3. **Code Style**: Follow PEP 8; use type hints where practical.
4. **Testing**: Add unit tests for any new functionality.
5. **Pull Request**: Submit a PR with a clear summary of changes.

---

## License and Citation

### License
This project is licensed under the **PolyForm Noncommercial License 1.0.0**. See the [LICENSE](LICENSE) file for full details.

This repository is provided for academic research, educational purposes, and technical evaluation (including peer review and talent assessment). Any integration into commercial production systems, trading platforms, or proprietary enterprise pipelines requires a separate commercial agreement.

If you are interested in commercial applications or adapting this architecture for enterprise use, please reach out directly:
* **Email:** dulanjanwijenayake@gmail.com
* **LinkedIn:** [Dulanjana Wijenayake](https://www.linkedin.com/in/dulanjan-wijenayake-58183a168)

### Academic Citation
If you utilize this architecture, codebase, or methodology in your research, please cite the corresponding thesis:

```bibtex
@mastersthesis{wijenayake2026miracle,
  author       = {Irosh Dulanjana Wijenayake Kankanamge},
  title        = {A Physics-Informed Hierarchical Learning Framework for Long-Horizon Photovoltaic Power Forecasting},
  year         = {2026},
  doi          = {10.5281/zenodo.20177800},
  url          = {https://zenodo.org/records/20177800}
}
```

**Disclaimer**: This software is provided "as-is" for research and evaluation purposes. The physics-informed layers are experimental implementations and do not constitute operational safety guarantees for grid infrastructure.
