import pandas as pd

# The Source of Truth file used by your inference script
SARNS_PATH = "freeze/final_thesis_v1/phase1_2024daily_final/rl/sarns_norm_with_blends.parquet"

try:
    print(f"🕵️  Auditing: {SARNS_PATH}")
    df = pd.read_parquet(SARNS_PATH)
    
    # We only care about the unique mapping between Action and Weights
    # We group by 'action' and take the mean (assuming they are constant per action)
    mapping = df.groupby("action")[["blend_short", "blend_long", "blend_physics"]].mean()
    
    print("\n--- 📝 THE OFFICIAL ACTION MAPPING ---")
    print(mapping)
    
    print("\n--- 🧠 HUMAN TRANSLATION ---")
    for action_id, row in mapping.iterrows():
        s, l, p = row["blend_short"], row["blend_long"], row["blend_physics"]
        
        # Logic to identify the strategy based on weights
        label = "UNKNOWN"
        if 0.48 <= s <= 0.50 and 0.25 <= l <= 0.27:
            label = "✅ BASELINE (Maintain)"
        elif l > 0.40:
            label = "📉 LONG-TERM BIAS (Strategic)"
        elif s > 0.52:
            label = "⚡ SHORT-TERM BIAS (Tactical)"
        elif p > 0.40:
            label = "☀️ PHYSICS BIAS (Clear Sky)"
            
        print(f"Action {action_id}: [Short={s:.2f}, Long={l:.2f}, Phys={p:.2f}] -> {label}")

except FileNotFoundError:
    print("❌ File not found. Check the path.")