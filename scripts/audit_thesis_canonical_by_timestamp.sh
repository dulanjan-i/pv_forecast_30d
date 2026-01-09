#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
from __future__ import annotations
from pathlib import Path
import datetime


def fmt(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')


def stat_line(p: Path) -> str:
    st = p.stat()
    return f"{fmt(st.st_mtime)}  {st.st_size:>10}  {p.as_posix()}"


def newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


headline_candidates = [
    Path('freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/text/results.md'),
    Path('freeze/final_thesis_v1/benchmarks/thesis_formatted_v3/tables/overall_metrics.csv'),
    Path('freeze/final_thesis_v1/eval/rq4_baseline_vs_policy/text/results.md'),
]

print('=== Thesis canonical timestamp audit (mtime, size, path) ===')
missing = [p for p in headline_candidates if not p.exists()]
if missing:
    print('Missing expected files:')
    for p in missing:
        print('  ', p)
    raise SystemExit(1)

for p in sorted(headline_candidates, key=lambda p: p.stat().st_mtime, reverse=True):
    print(stat_line(p))

print('\n=== Newest benchmark-suite summary under freeze/final_thesis_v1/benchmarks ===')
bench_summaries = list(Path('freeze/final_thesis_v1/benchmarks').glob('**/text/results.md'))
if bench_summaries:
    p = newest(bench_summaries)
    print(stat_line(p))
else:
    print('No benchmark summaries found.')

print('\n=== Newest rq4_baseline_vs_policy summary under freeze/final_thesis_v1 ===')
rq4_summaries = list(Path('freeze/final_thesis_v1').glob('**/rq4_baseline_vs_policy/**/text/results.md'))
if rq4_summaries:
    p = newest(rq4_summaries)
    print(stat_line(p))
else:
    print('No RQ4 summaries found.')
PY
