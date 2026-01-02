#!/usr/bin/env bash
set -euo pipefail

# If htop is available, use it. Otherwise fall back to top.
if command -v htop >/dev/null 2>&1; then
  htop
else
  top
fi
