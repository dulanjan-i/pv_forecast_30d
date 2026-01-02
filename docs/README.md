# MiRACLE Documentation

This folder contains technical documentation, setup guides, and archived audit reports.

## Structure

### `/technical/` - Core Technical Documentation
- **MIRACLE_SCIENTIFIC_WORKFLOW.md** - Academic thesis-ready workflow (1300+ lines)
- **MIRACLE_SAR_SPACE_CLEAN.md** - RL State-Action-Reward space technical spec
- **SAR_SPACE_COMPARISON.md** - Before/after refactor comparison (4 DDQN → 1+3)
- **REFACTOR_SUMMARY.md** - RL architecture refactor rationale
- **MIRACLE_EXPLAINED_LIKE_TODDLER.md** - Accessible stakeholder guide
- **PHYSICS_CONSTRAINED_INFERENCE_DETAILED.md** - Physics-aware forecasting
- **PHYSICS_GLUE_IMPLEMENTATION.md** - PVLib integration details

### `/archive/` - Completed Audits & Status Reports
- AUDIT_LSTM_PRETRAIN.md
- CHECKPOINT_VERIFICATION_REPORT.md
- HIERARCHICAL_ARCHITECTURE_AUDIT.md
- TFT_ERROR_AUDIT.md
- TFT_INTEGRATION_*.md
- WEATHER_API_*.md
- V1.0_FINAL_TFT_MIGRATION_COMPLETE.md

### `/` (root) - Setup & Configuration
- **DBFZ_HPC_README.md** - HPC cluster setup guide
- **INSTALL_CALC02.md** - Calc02 server installation
- **README_CALC02.md** - Calc02 usage instructions
- **README_LSTM.md** - LSTM encoder documentation
- **TRAINING_CONFIG_SUMMARY.md** - Model training configs
- **MIRACLE_V1_TODO.md** - Version 1.0 task list
- **TODO_TFT_pipeline.md** - TFT pipeline tasks
- **PIPELINE_LOCK_STATUS.md** - Pipeline component locks

## Quick Links

**For Thesis/Publication:**
- [Scientific Workflow](technical/MIRACLE_SCIENTIFIC_WORKFLOW.md) - Complete academic documentation
- [SAR Space](technical/MIRACLE_SAR_SPACE_CLEAN.md) - RL design with code references

**For Stakeholders:**
- [Explained Like Toddler](technical/MIRACLE_EXPLAINED_LIKE_TODDLER.md) - Accessible overview

**For Developers:**
- [Refactor Summary](technical/REFACTOR_SUMMARY.md) - Why we moved to 1 DDQN + 3 advisors
- [Physics Glue](technical/PHYSICS_GLUE_IMPLEMENTATION.md) - How PVLib integrates

**For HPC Users:**
- [DBFZ HPC README](DBFZ_HPC_README.md) - Cluster job submission
- [Install Guide](INSTALL_CALC02.md) - Server setup
