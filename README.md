# MiRACLE v1.0: Physics-Informed Hierarchical Learning for Long-Horizon Photovoltaic Power Forecasting

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20177800.svg)](https://doi.org/10.5281/zenodo.20177800)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This repository contains the advanced **dual-head Temporal Fusion Transformer (TFT)** engine developed for my Master's thesis. It is made available for academic research, peer review, and technical evaluation.

![MiRACLE Architecture](architecture%20diagrams/core_architecture.html)

## Abstract
This thesis presents MiRACLE v1.0, a physics-informed hierarchical learning framework designed to solve the complex challenge of long-horizon (30-day ahead at 15-minute intervals) photovoltaic power forecasting. By utilizing a novel dual-head Temporal Fusion Transformer (TFT) architecture governed by a Double Deep Q-Network (DDQN) meta-controller, this framework forcefully fuses short-horizon ramp accuracy with long-horizon shape fidelity.

## Key Features
* **Dual-Head TFT Architecture**: Parallel independent instances structured to achieve cooperative fusion of outputs.
* **DDQN Meta-Controller**: Dynamically adapts weighting parameters to changing environments and market conditions.
* **Physics Glue & Constraints**: Integrates physical constraints directly into the learning pipeline to prevent operational anomalies.
* **Robust Data Engineering**: Engineered to handle complex, messy, real-world data pipelines.

---

## Support & Documentation
* **Full Thesis**: See (https://zenodo.org/records/20177801) for complete methodology, literature review, and detailed results.
* **Data Schema**: [docs/final_data_cols.md](docs/final_data_cols.md)
* **Physics Glue**: [docs/glue.md](docs/glue.md)
* **Issues**: Open a GitHub issue or contact the author.

---

## Contributing
We welcome contributions! Please follow these guidelines:
1. **Issue**: Open an issue describing the bug or feature request.
2. **Fork & Branch**: Create a feature branch from `main`.
3. **Code Style**: Follow PEP 8; use type hints where practical.
4. **Testing**: Add unit tests for any new functionality.
5. **Documentation**: Update docstrings and this README if needed.
6. **Pull Request**: Submit a PR with a clear summary of changes.

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
**Disclaimer**: This software is provided "as-is" for research and evaluation purposes. The physical constraints and physics-informed layers are experimental implementations and do not constitute operational safety guarantees for physical grid infrastructure. Before deploying any concepts in a grid-critical context, validate against operational needs.

Last Updated: May 2026

Status: Frozen v1.0 (Master's Thesis Artifact)

Maintenance: Limited to critical bug fixes; new features are out of scope for v1.0.
