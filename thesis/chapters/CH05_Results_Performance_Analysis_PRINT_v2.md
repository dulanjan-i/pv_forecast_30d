# Chapter 5 — Results and Performance Analysis (Print-Ready)

## 5.1 Canonical evaluation framing

All headline quantitative claims in this chapter are based on a temporally isolated backtesting protocol on plant_03, where the evaluation period is drawn from a future year that was never used during model fitting, preprocessing parameter estimation, or hyperparameter selection. This design is essential for PV forecasting because even small amounts of temporal leakage can lead to optimistic performance estimates. The evaluation is performed at 15-minute resolution over the full 30-day horizon and is intended to approximate deployment conditions.

To focus evaluation on operationally relevant periods, the metrics exclude near-zero nighttime points using a small threshold on the true capacity-normalized output. This avoids trivial inflation of performance by long nighttime intervals.

## 5.2 Metrics

Let y_i denote the true capacity-normalized PV output and let y-hat_i denote the predicted value. The core metrics are defined as follows.

$$
\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| y_{i} - \hat{y}_{i} \right|
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( y_{i} - \hat{y}_{i} \right)^{2}}
$$

$$
\mathrm{MBE} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_{i} - y_{i} \right)
$$

$$
R^{2} = 1 - \frac{\sum_{i=1}^{N} \left( y_{i} - \hat{y}_{i} \right)^{2}}{\sum_{i=1}^{N} \left( y_{i} - \bar{y} \right)^{2}}
$$

## 5.3 Overall accuracy and ablation outcomes

Under the canonical backtesting protocol, the integrated MiRACLE configuration achieves the strongest overall accuracy among the evaluated baselines. The MiRACLE configuration attains an RMSE of 0.11713 on the filtered evaluation set. This improves upon a model-only forecaster without physics-informed integration (RMSE 0.140186) and a physics-only baseline (RMSE 0.163976). Horizon-isolated variants perform substantially worse, with the short-horizon-only variant attaining RMSE 0.167144 and the long-horizon-only variant attaining RMSE 0.223948.

These results support the methodological claim that MiRACLE’s gains arise from complementary contributions: physics-informed covariates provide a strong inductive bias and plausibility support, while learned models capture complex interactions and latent temporal effects not fully represented by physics alone.

## 5.4 Horizon-dependent performance

Forecasting difficulty varies strongly with lead time. The near term is sensitive to short-lived meteorological fluctuations and persistence in recent PV behavior, while longer horizons are increasingly constrained by weather forecast uncertainty and seasonal structure. MiRACLE demonstrates consistent performance across the horizon buckets used in this thesis. The RMSE values are 0.118582 for 0–24 hours, 0.119244 for 2–7 days, and 0.116504 for 8–30 days.

The persistence of performance advantages across horizon buckets indicates that MiRACLE does not merely overfit the near-term regime. Instead, the combination of temporal embeddings, physics-informed features, and hierarchical inference yields systematic improvements throughout the 30-day horizon.

## 5.5 Transfer learning effects

Warm-start transfer learning yields modest but consistent improvements over a cold-start configuration under the canonical evaluation. The warm-start configuration attains RMSE 0.11713, while the cold-start configuration attains RMSE 0.119183. This difference supports the hypothesis that pretrained temporal representations improve stability and reduce error under domain shift, particularly when data completeness and operational regimes differ from the pretraining domain.

## 5.6 Reinforcement-learning meta-controller evaluation

The reinforcement-learning meta-controller is evaluated against a baseline decision rule under the same strict backtesting conditions. Under the canonical evaluation, the learned policy is essentially neutral in overall RMSE relative to baseline, and it performs slightly worse in the 0–24 hour bucket. These outcomes indicate that the present controller configuration does not yet produce consistent gains over a strong baseline.

This result should be interpreted as an informative negative finding rather than as evidence against the architectural idea. The evaluation protocol is sensitive to small deltas and establishes a reproducible framework for future refinement of state representation, reward shaping, and training environment realism.

## 5.7 Summary

The results support the thesis claim that MiRACLE’s hybrid architecture improves long-horizon PV forecasting accuracy relative to model-only and physics-only baselines. The gains persist across lead-time buckets, indicating robust improvements rather than a narrow near-term effect. Transfer learning provides a small but consistent advantage, and the reinforcement-learning component establishes an extensible framework for adaptive operational decision-making, even though the evaluated policy is near-neutral under the current configuration.
