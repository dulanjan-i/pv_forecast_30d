import pandas as pd
import sys

# CHANGE THIS to the path of the file you are currently generating
FILE_PATH = "freeze/final_thesis_v1/phase1_2024daily_final/processed/predictions_phase1_policy_rerun.parquet"

try:
    print(f"🔍 Inspecting: {FILE_PATH} ...")
    df = pd.read_parquet(FILE_PATH)
    
    print(f"✅ Status: LOADED. Rows: {len(df):,}")
    print(f"📅 Date Range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    
    print("\n--- 🕵️ ACTION ANALYSIS (The Truth) ---")
    # Group by Action and show the average weights used
    stats = df.groupby("policy_action")[["blend_short", "blend_long", "blend_physics"]].mean()
    stats["count"] = df["policy_action"].value_counts()
    print(stats)
    
    print("\n--- 🧪 DIAGNOSIS ---")
    if 1 in stats.index:
        short_weight = stats.loc[1, "blend_short"]
        if abs(short_weight - 0.4875) < 0.01:
            print("🔒 RESULT: ORIGINAL / CONSERVATIVE RUN")
            print("   Action 1 is 'Safe' (Mapped to Baseline Weights).")
            print("   -> Use this for the 'Stability' Thesis narrative.")
        elif abs(short_weight - 0.55) < 0.01:
            print("🔓 RESULT: UNLOCKED / FIXED RUN")
            print("   Action 1 is 'Active' (0.55 Short Bias).")
            print("   -> Use this for the 'Surgical Optimization' narrative.")
        else:
            print(f"⚠️ RESULT: CUSTOM ({short_weight})")
    else:
        print("⚠️ Action 1 NOT FOUND in dataset.")

except FileNotFoundError:
    print("❌ File not found yet. The run might still be initializing.")
except Exception as e:
    print(f"❌ Error: {e}")