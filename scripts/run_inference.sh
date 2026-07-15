#!/usr/bin/env bash
# =============================================================================
# MiRACLE v1.0 Inference Runner
# =============================================================================
# This script is the main entrypoint for the Docker container.
# It wraps the Python inference pipeline with a friendly CLI.
#
# Usage inside Docker:
#   docker run miracle-inference:v1.0 /app/scripts/run_inference.sh \
#     --date 2026-01-02 \
#     --output /app/outputs/forecast.parquet
#
# Usage directly (dev):
#   bash scripts/run_inference.sh --date 2026-01-02
# =============================================================================

set -euo pipefail  # exit on error, undefined var, pipe failure

# ── Defaults ─────────────────────────────────────────────────────────────────
DATE="${MIRACLE_FORECAST_DATE:-$(date +%Y-%m-%d)}"
OUTPUT_DIR="${MIRACLE_OUTPUT_DIR:-/app/outputs}"
RL_CKPT="${MIRACLE_RL_CKPT:-/app/checkpoints/rl_v2/ddqn_best.pt}"
SHORT_CKPT="${MIRACLE_SHORT_CKPT:-/app/V1.0_FINAL_TFT/shorthead_seed42/checkpoints/best.pt}"
LONG_CKPT="${MIRACLE_LONG_CKPT:-/app/V1.0_FINAL_TFT/longhead_seed43/checkpoints/best.pt}"
PLANT_META="${MIRACLE_PLANT_META:-/app/V1.0_FINAL_TFT/plant_metadata/plant_03.json}"

# ── Parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --date)        DATE="$2";       shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --rl-ckpt)     RL_CKPT="$2";    shift 2 ;;
        --help|-h)
            echo "Usage: run_inference.sh [--date YYYY-MM-DD] [--output-dir /path]"
            echo ""
            echo "Options:"
            echo "  --date        Forecast start date (default: today)"
            echo "  --output-dir  Directory to write forecast parquet (default: /app/outputs)"
            echo "  --rl-ckpt     Override RL checkpoint path"
            echo ""
            echo "Environment variables (alternative to flags):"
            echo "  MIRACLE_FORECAST_DATE, MIRACLE_OUTPUT_DIR, MIRACLE_RL_CKPT"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/forecast_${DATE}.parquet"

echo "=============================================="
echo " MiRACLE v1.0 — 30-Day PV Forecast"
echo "=============================================="
echo "  Forecast date : $DATE"
echo "  Output file   : $OUTPUT_FILE"
echo "  RL checkpoint : $RL_CKPT"
echo "  Short-head    : $SHORT_CKPT"
echo "  Long-head     : $LONG_CKPT"
echo "=============================================="

# ── Verify checkpoints (SHA256 spot-check) ───────────────────────────────────
echo ""
echo "Verifying checkpoint integrity..."
python - << PYCHECK
import hashlib, sys
def sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(65536), b''): h.update(c)
    return h.hexdigest()

checks = {
    "$SHORT_CKPT": "5d9568f624258d5db46d0b346fee95f3c4fa6da28e224a1fdd95f17808944137",
    "$LONG_CKPT":  "1a21bc37ee7aef146756cceb64992d91fdd8d7fafd4e650c96cbd494fc66968e",
    "$RL_CKPT":    "bcc32ceb87765fe5dd90248e639619b17fd1a137b2592f8cd431cac03f75e549",
}
ok = True
for path, expected in checks.items():
    actual = sha256(path)
    status = "OK" if actual == expected else "MISMATCH"
    print(f"  {status}: {path.split('/')[-1]}")
    if actual != expected:
        print(f"    expected: {expected}")
        print(f"    got:      {actual}")
        ok = False
if not ok:
    print("CHECKPOINT INTEGRITY FAILED — aborting")
    sys.exit(1)
print("All checksums verified.")
PYCHECK

# ── Run inference ─────────────────────────────────────────────────────────────
echo ""
echo "Running inference pipeline..."
python -m src.inference.physics_aware_forecaster \
    --forecast-start  "$DATE" \
    --short-ckpt      "$SHORT_CKPT" \
    --long-ckpt       "$LONG_CKPT" \
    --plant-meta      "$PLANT_META" \
    --rl-ckpt         "$RL_CKPT" \
    --use-live-weather \
    --output-file     "$OUTPUT_FILE"

echo ""
echo "Done. Forecast written to: $OUTPUT_FILE"
