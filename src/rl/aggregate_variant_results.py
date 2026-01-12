#!/usr/bin/env python3
"""Aggregate per-variant result parquets into a summary CSV.

Writes: experiments/rl/counterfactuals/plant_03/summary/variant_summary.csv
"""
from pathlib import Path
import pandas as pd


def main():
    res_dir = Path("experiments/rl/counterfactuals/plant_03/results")
    out_dir = Path("experiments/rl/counterfactuals/plant_03/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(res_dir.glob("results_*.parquet"))
    if not files:
        print("No result files found in", res_dir)
        return

    rows = []
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print("Skipping", p, "(read error)", e)
            continue

        variant = p.stem.replace("results_", "")
        overall_mean = float(df["rmse_day1"].mean()) if "rmse_day1" in df.columns else None
        count = len(df)

        # mean per action
        action_means = df.groupby("action")["rmse_day1"].mean().to_dict() if "action" in df.columns else {}

        row = {"variant": variant, "file": str(p), "rows": count, "rmse_mean": overall_mean}
        # add action means as columns action_0..action_9
        for a in range(10):
            row[f"action_{a}_mean"] = float(action_means.get(a, float("nan")))

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("variant")
    out_path = out_dir / "variant_summary.csv"
    out_df.to_csv(out_path, index=False)
    print("WROTE:", out_path)

    # Also write a tiny markdown summary
    md = out_dir / "summary.md"
    with open(md, "w") as fh:
        fh.write("# Variant summary\n\n")
        fh.write(f"Files aggregated: {len(out_df)}\n\n")
        fh.write("| variant | rows | rmse_mean | best_action | best_action_rmse |\n")
        fh.write("|---|---:|---:|---:|---:|\n")
        for _, r in out_df.iterrows():
            action_cols = [c for c in out_df.columns if c.startswith("action_")]
            best_act = None
            best_val = None
            for c in action_cols:
                v = r[c]
                if pd.isna(v):
                    continue
                if best_val is None or v < best_val:
                    best_val = float(v)
                    best_act = c.replace("action_", "").replace("_mean", "")

            fh.write(f"| {r['variant']} | {int(r['rows'])} | {r['rmse_mean']:.4f} | {best_act or '-'} | {best_val if best_val is not None else '-'} |\n")

    print("WROTE:", md)


if __name__ == '__main__':
    main()
