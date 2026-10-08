#!/usr/bin/env bash
set -euo pipefail

# Live GPU stats, refresh every 1s
# Works on calc02 and compute nodes if nvidia-smi exists.

INTERVAL="${1:-1}"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l "$INTERVAL"
