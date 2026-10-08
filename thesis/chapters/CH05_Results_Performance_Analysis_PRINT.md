# Chapter 5 — Results and Performance Analysis (Print-Ready Draft)

## 5.1 Headline performance and the backtesting protocol

All headline quantitative results in this chapter are derived from a temporally isolated backtesting protocol on an unseen future period for the target plant. The experimental design reserves the year 2024 for out-of-sample evaluation on plant_03 and ensures that the training phase does not incorporate any 2024 observations from the test plant. This distinction is crucial because PV forecasting is especially vulnerable to optimistic evaluation if future data influence model fitting, preprocessing statistics, or hyperparameter choices.

Backtesting is conducted under operationally realistic conditions. Forecasts are generated across a rolling set of forecast origins and evaluated at 15-minute resolution over the full 30-day horizon. To avoid trivial inflation of performance by nighttime zeros, evaluation excludes points where the true normalized power is below a small threshold. This focuses metrics on daylight production where forecasting accuracy is operationally meaningful.

## 5.2 Overall forecasting accuracy

The integrated MiRACLE configuration achieves the strongest overall performance among the evaluated baselines under the canonical backtesting protocol. In terms of root mean squared error on the filtered daytime set, the MiRACLE configuration attains an RMSE of 0.11713. This result represents a substantial improvement over a forecaster without physics-informed integration, which attains an RMSE of 0.140186, and over a physics-only baseline, which attains an RMSE of 0.163976. Single-component variants that rely on only one horizon-specific forecaster exhibit materially worse performance, with the short-horizon-only variant attaining an RMSE of 0.167144 and the long-horizon-only variant attaining an RMSE of 0.223948.

These results support two conclusions. First, physics-informed integration provides a meaningful inductive bias that reduces error relative to a purely data-driven forecaster. Second, the combination of horizon-specific modeling and integration is necessary for robust long-horizon performance; single-horizon variants fail to capture the full structure of the 30-day forecasting problem.

## 5.3 Horizon-dependent performance

PV forecasting difficulty varies strongly with lead time. In the near term (0–24 hours), accuracy depends on short-lived meteorological variability and immediate persistence in recent PV behavior. At longer horizons (days to weeks), accuracy is increasingly constrained by the uncertainty of weather forecasts and by slow seasonal structure. Under the canonical backtest, MiRACLE exhibits consistent performance across the horizon buckets, with RMSE values of 0.118582 for 0–24 hours, 0.119244 for 2–7 days, and 0.116504 for 8–30 days.

The persistence of the performance advantage across horizon buckets is notable because it indicates that MiRACLE’s improvements are not restricted to short-term calibration. Instead, the architecture appears to provide systematic gains in both near-term variability handling and longer-term structural forecasting.

## 5.4 Contribution of physics-informed features

Physics-only modeling provides a meaningful baseline for PV generation because it encodes deterministic solar geometry and plausible irradiance structure. However, physics-only predictions do not fully account for site-specific operational characteristics, meteorological uncertainty, and latent temporal effects. The observed gap between the physics-only baseline (RMSE 0.163976) and the integrated MiRACLE configuration (RMSE 0.11713) demonstrates that physics alone is insufficient for high-accuracy operational forecasting in this setting.

At the same time, the gap between the forecaster without physics integration (RMSE 0.140186) and MiRACLE suggests that physics-informed features and constraints materially improve the learned model’s performance. This finding is consistent with the broader forecasting literature: physically grounded covariates reduce the hypothesis space a neural model must learn, improving both sample efficiency and robustness.

## 5.5 Transfer learning and warm-start effects

A central hypothesis of the methodology is that representation learning and warm-starting reduce error under distribution shift. Under the canonical evaluation, the warm-start configuration attains an RMSE of 0.11713, while a cold-start configuration attains an RMSE of 0.119183. Although the absolute difference is modest, it is consistent with the expectation that warm initialization provides more stable behavior, particularly in the near term where local calibration matters.

The importance of warm-starting is not only the overall delta but also the robustness implication: warm-starting reduces sensitivity to optimization instabilities and can improve performance in data-sparse regimes. This is especially relevant given the heterogeneous completeness of raw plant exports and the need to rely on disciplined training splits.

## 5.6 Reinforcement-learning meta-controller evaluation

The reinforcement-learning meta-controller is evaluated against a baseline decision rule under the same backtesting framework. Under the canonical evaluation, the policy is essentially neutral in overall RMSE, with a very small delta relative to baseline. In the 0–24 hour bucket, the policy performs slightly worse than baseline under the evaluated configuration.

This outcome suggests that the current policy, as trained and evaluated in this thesis, does not yet deliver consistent improvements over a strong baseline. However, the neutral result is informative: it indicates that the evaluation protocol is sensitive enough to detect small deltas and that the architecture can support rigorous future exploration. The discussion chapter interprets this finding as evidence that additional work is required to improve state representation, reward shaping, and environment realism to enable robust policy gains.

## 5.7 Summary

The canonical backtesting results support the thesis claim that MiRACLE’s integrated architecture improves long-horizon PV forecasting accuracy relative to both physics-only and model-only baselines. The gains persist across horizon buckets, suggesting that the combination of physics-informed features, learned temporal embeddings, and multi-horizon forecasting provides complementary benefits. Transfer learning yields modest but consistent improvements, and the reinforcement-learning component establishes a framework for future adaptive control, even though the evaluated policy is near-neutral under the present configuration.
