================================================================================
🎯 GROUND TRUTH VALIDATION: Phase 1 vs Real Measurements (2024)
================================================================================

📂 Loading ground truth data (15-min aligned)...
   Loaded: 35,136 timesteps (15-min resolution)
   Range: 2024-01-01 00:15:00 → 2025-01-01 00:00:00
   Max power: 5901.6 kW
   Max normalized: 0.8020

📂 Loading Phase 1 predictions...
   Loaded: 152,640 timesteps
   Range: 2023-12-01 00:00:00 → 2024-12-28 23:45:00

🔗 Merging predictions with ground truth...
   ✅ Matched timesteps: 144,572
   Coverage: 94.7% of predictions
   Overlap: 2024-01-01 00:15:00 → 2024-12-28 23:45:00

================================================================================
📊 ERROR METRICS
================================================================================

Valid comparisons: 124,032 timesteps

📉 OVERALL PERFORMANCE:
   RMSE (normalized):  0.13886  (13.89% of capacity)
   RMSE (absolute):    1021.9 kW
   MAE (normalized):   0.07547  (7.55% of capacity)
   MAE (absolute):     555.4 kW
   R² Score:           0.3591
   MAPE:               1328912.22%

📊 COMPARISON TO PSEUDO-RMSE:
   Pseudo-RMSE (forecast disagreement): 0.00693
   True RMSE (vs ground truth):         0.13886
   Ratio (True / Pseudo):                20.04×
   ⚠️  Pseudo-RMSE significantly underestimated true error

📈 BIAS ANALYSIS:
   Mean bias:         +0.01829 (+1.83%)
   Median bias:       +0.00000
   Std dev:           0.13765
   Over-predictions:  38,716 / 124,032 (31.2%)
   ⚠️  MODEL OVERESTIMATES by 135 kW on average

🌞 DAYLIGHT PERFORMANCE (06:00-18:00):
   Samples: 67,080
   RMSE: 0.18828 (1385.5 kW)
   MAE:  0.13593 (1000.3 kW)
   R²:   0.1163

================================================================================
📅 MONTHLY ERROR BREAKDOWN
================================================================================

Month      Samples    RMSE       MAE        R²       Bias      
----------------------------------------------------------------------
2024-01    12,668     0.06778    0.02711    0.6902   -0.01218  
2024-02    11,904     0.07573    0.03269    0.6569   -0.01126  
2024-03    12,844     0.12321    0.05984    0.4933   +0.00719  
2024-04    12,288     0.16297    0.09650    0.2218   +0.02736  
2024-05    12,768     0.17714    0.10475    0.0840   +0.04475  
2024-06    12,384     0.15247    0.09704    0.4467   +0.01873  
2024-07    12,672     0.18486    0.11445    0.0578   +0.05227  
2024-08    12,864     0.14734    0.09295    0.4620   +0.01414  
2024-09    12,288     0.13576    0.07436    0.3606   +0.01624  
2024-10    11,152     0.10223    0.05135    -0.1433  +0.02540  
2024-11    130        0.01311    0.01216    -6.1175  -0.01216  
2024-12    70         0.01400    0.01336    -10.2984 -0.01336  

📊 DAILY ERROR ANALYSIS:

🏆 BEST 5 DAYS (Lowest RMSE):
   2024-01-02: RMSE = 0.00691 (50.9 kW)
   2024-01-08: RMSE = 0.01090 (80.2 kW)
   2024-01-06: RMSE = 0.01267 (93.3 kW)
   2024-01-13: RMSE = 0.01324 (97.5 kW)
   2024-10-23: RMSE = 0.01772 (130.4 kW)

💥 WORST 5 DAYS (Highest RMSE):
   2024-05-13: RMSE = 0.28341 (2085.6 kW)
   2024-05-20: RMSE = 0.28364 (2087.3 kW)
   2024-04-27: RMSE = 0.29137 (2144.1 kW)
   2024-05-26: RMSE = 0.29477 (2169.2 kW)
   2024-07-09: RMSE = 0.31798 (2339.9 kW)

================================================================================
🎯 FINAL VERDICT
================================================================================

   Grade: B GOOD
   RMSE: 0.13886 (13.89% of capacity) = 1021.9 kW
   R²: 0.3591
   Verdict: Acceptable, room for optimization

✅ Saved validation data to: data/processed/test_phase1_dec2023_dec2024/validation_vs_ground_truth.parquet

================================================================================