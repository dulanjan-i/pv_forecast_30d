#!/usr/bin/env python3
"""Render selected PlantUML diagrams to thesis-ready PNG + PDF.

Why this exists:
- Local `plantuml` CLI / SVG->PDF converters may be missing.
- Java may exist but a PlantUML jar may not.

Approach:
- Uses the public PlantUML server to render PNG.
- Wraps the PNG in a single-page PDF via reportlab.

Outputs:
- thesis/figures/architecture/<name>.png
- thesis/figures/architecture/<name>.pdf

If you cannot use network access, you can adapt this script to use a local
PlantUML jar: `java -jar plantuml.jar -tpng`.
"""

from __future__ import annotations

import argparse
import base64
import os
import pathlib
import sys
import urllib.error
import urllib.request
import zlib

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def _plantuml_encode(text: str) -> str:
    """Encode PlantUML text using the server's deflate + custom base64."""

    compressed = zlib.compress(text.encode("utf-8"), level=9)
    # PlantUML expects raw DEFLATE stream (no zlib headers)
    raw_deflate = compressed[2:-4]
    encoded = base64.b64encode(raw_deflate).decode("ascii")

    # Translate standard base64 into PlantUML's encoding alphabet.
    # See PlantUML server encoding.
    return (
        encoded.replace("+", "-")
        .replace("/", "_")
        .replace("=", "")
    )


def _fetch(url: str, timeout_s: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pv_forecast_30d/plantuml-render"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _write_pdf_from_png(png_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    # Use landscape A4; this works reasonably well for pipeline diagrams.
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))

    # Leave margins; fit image preserving aspect ratio.
    margin = 0.5 * inch
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    from reportlab.lib.utils import ImageReader

    img = ImageReader(str(png_path))
    img_w, img_h = img.getSize()

    scale = min(avail_w / img_w, avail_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2
    c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()


def render_puml(
    puml_path: pathlib.Path,
    out_dir: pathlib.Path,
    server_base: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    text = puml_path.read_text(encoding="utf-8")

    encoded = _plantuml_encode(text)
    base = server_base.rstrip("/")
    png_url = f"{base}/png/{encoded}"

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / (puml_path.stem + ".png")
    pdf_path = out_dir / (puml_path.stem + ".pdf")

    png_bytes = _fetch(png_url)
    png_path.write_bytes(png_bytes)
    _write_pdf_from_png(png_path, pdf_path)

    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        default=os.environ.get("PLANTUML_SERVER", "https://www.plantuml.com/plantuml"),
        help="PlantUML server base URL (default: https://www.plantuml.com/plantuml)",
    )
    parser.add_argument(
        "--out-dir",
        default="thesis/figures/architecture",
        help="Output directory for rendered figures",
    )
    parser.add_argument(
        "puml",
        nargs="*",
        help="One or more .puml files to render",
    )

    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir

    if args.puml:
        inputs = [pathlib.Path(p) for p in args.puml]
    else:
        inputs = [
            repo_root / "architecture diagrams" / "miracle_high_level.puml",
            repo_root / "architecture diagrams" / "miracle_full_data_pipeline.puml",
        ]

    failures: list[str] = []
    for p in inputs:
        p = (repo_root / p) if not p.is_absolute() else p
        if not p.exists():
            failures.append(f"Missing file: {p}")
            continue

        try:
            png_path, pdf_path = render_puml(p, out_dir=out_dir, server_base=args.server)
            print(f"Rendered: {p} -> {png_path.relative_to(repo_root)} and {pdf_path.relative_to(repo_root)}")
        except urllib.error.URLError as e:
            failures.append(f"Network/URL error for {p}: {e}")
        except Exception as e:
            failures.append(f"Failed for {p}: {e}")

    if failures:
        print("\nErrors:")
        for f in failures:
            print(f"- {f}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
