#!/usr/bin/env python3
"""Bundle rendered architecture figures into a zip for reliable transfer.

VS Code Remote's per-file download sometimes results in users accidentally
saving/opening a PlantUML server URL (HTML error page) rather than the local
rendered artifacts. A single zip reduces this risk.

Outputs:
- thesis/figures/architecture/miracle_architecture_figures.zip

Usage:
  /path/to/python scripts/bundle_architecture_figures.py
"""

from __future__ import annotations

import pathlib
import zipfile


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    arch_dir = repo_root / "thesis" / "figures" / "architecture"

    inputs = [
        arch_dir / "miracle_high_level.png",
        arch_dir / "miracle_high_level.pdf",
        arch_dir / "miracle_full_data_pipeline.png",
        arch_dir / "miracle_full_data_pipeline.pdf",
        arch_dir / "SHA256SUMS.txt",
    ]

    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    out_zip = arch_dir / "miracle_architecture_figures.zip"
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in inputs:
            zf.write(p, arcname=p.name)

    print(f"Wrote: {out_zip.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
