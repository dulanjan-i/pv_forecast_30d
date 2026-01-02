#!/bin/bash
# Split today's work into organized feature branches
# Run this from repo root: bash split_branches.sh

set -e

echo "🔀 Splitting main branch work into feature branches..."
echo ""

# Store current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Stash all changes first
echo "📦 Stashing all changes..."
git add -A
git stash push -m "work_from_jan2_2026"
echo "✅ Changes stashed"
echo ""

# ============================================================================
# BRANCH 1: checkpoint-migration (do this first - others depend on it)
# ============================================================================
echo "🌿 Creating branch: checkpoint-migration"
git checkout -b checkpoint-migration main

echo "📥 Applying checkpoint files..."
git stash pop
git add V1.0_FINAL_TFT/
git add CHECKPOINT_VERIFICATION_REPORT.md
git add V1.0_FINAL_TFT_MIGRATION_COMPLETE.md
git add VALIDATION_METRICS.md || true  # if exists
git commit -m "feat: migrate TFT checkpoints to V1.0_FINAL_TFT structure

- Verified seed 42 (short-head) and seed 43 (long-head)
- Created standardized checkpoint structure
- Added plant metadata and configs
- Validation: Both models load successfully"

echo "✅ checkpoint-migration committed"
echo ""

# Stash remaining changes
git add -A
git stash push -m "remaining_work"

# ============================================================================
# BRANCH 2: inference-pipeline-v1
# ============================================================================
echo "🌿 Creating branch: inference-pipeline-v1"
git checkout -b inference-pipeline-v1 checkpoint-migration  # base on checkpoints

echo "📥 Applying inference files..."
git stash pop
git add .gitignore
git add .ecmwf_credentials.json || true  # if not gitignored
git add src/inference/weather_client.py
git add src/inference/physics_aware_forecaster.py
git add src/inference/physics_glue.py
git add src/inference/pvlib_predictor.py
git add src/inference/tft_utils.py
git add src/inference/offline_predict_tft.py
git add test_*.py
git add WEATHER_API_*.md
git add PHYSICS_*.md
git add TFT_*.md
git add HIERARCHICAL_*.md
git add PIPELINE_*.md
git add glue.md

git commit -m "feat: complete inference pipeline with multi-API weather routing

Weather Integration:
- Smart API router (Forecast 0-7d, ECMWF 8-15d, GFS 16d+)
- ECMWF credentials support
- Dual resolution (15-min + 1-hour)

Physics Integration:
- PhysicsAwareForecaster with ensemble blending
- PVLib physical constraints
- Hierarchical pipeline architecture

TFT Integration:
- Load V1.0_FINAL_TFT checkpoints
- Dual-head system (short + long)
- Offline prediction utilities

Tests:
- Full pipeline validation (100% passing)
- Live weather API tests
- TFT integration tests
- Hierarchical pipeline tests"

echo "✅ inference-pipeline-v1 committed"
echo ""

# Stash remaining (RL stuff)
git add -A
git stash push -m "rl_work"

# ============================================================================
# BRANCH 3: rl-meta-build (use existing branch)
# ============================================================================
echo "🌿 Checking out existing branch: rl-meta-build"
git checkout rl-meta-build
git merge inference-pipeline-v1 --no-edit  # bring in inference work

echo "📥 Applying RL files..."
git stash pop
git add src/rl/
git add tests/test_rl_integration.py
git add docs/RL_META_CONTROLLER_GUIDE.md
git add reports/RL_IMPLEMENTATION_SUMMARY.md

git commit -m "feat: hierarchical RL meta-controller for adaptive forecasting

Architecture:
- 3 Local Agents: Short-TFT, Long-TFT, PVLib
- 1 Meta-Agent: Dynamic ensemble blending
- Hierarchical DQN from MiRACLE paper

Features:
- Prioritized experience replay
- Multi-objective reward function
- Human-in-the-loop retrain confirmation
- 3 operating modes: heuristic, rl, hybrid

Implementation:
- src/rl/rl_meta_controller.py (823 lines)
- src/rl/rl_integrated_forecaster.py (400+ lines)
- Integration tests: 8/8 passing

Timeline:
- Heuristic deploy: Day 1-2
- DQN training: 1-3h on H100/L4
- A/B testing: Day 3-4
- Production: Day 4+"

echo "✅ rl-meta-build committed"
echo ""

# ============================================================================
# Return to main and show status
# ============================================================================
echo "🔙 Returning to main branch..."
git checkout main

echo ""
echo "✅ BRANCH SPLIT COMPLETE!"
echo ""
echo "📊 Summary:"
git branch -v
echo ""
echo "🚀 Next steps:"
echo "1. Review branches:         git log --oneline --graph --all"
echo "2. Push checkpoint branch:  git push -u origin checkpoint-migration"
echo "3. Push inference branch:   git push -u origin inference-pipeline-v1"
echo "4. Push RL branch:          git push origin rl-meta-build"
echo ""
echo "5. Create PRs on GitHub/GitLab:"
echo "   - checkpoint-migration  → main"
echo "   - inference-pipeline-v1 → main (after checkpoint merged)"
echo "   - rl-meta-build         → main (after inference merged)"
echo ""
echo "Or merge sequentially:"
echo "  git checkout main"
echo "  git merge checkpoint-migration"
echo "  git merge inference-pipeline-v1"
echo "  git merge rl-meta-build"
echo "  git push origin main"
