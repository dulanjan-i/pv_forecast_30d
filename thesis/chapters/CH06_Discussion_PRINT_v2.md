# Chapter 6 — Discussion (Print-Ready)

## 6.1 Interpretation of the main findings

The results demonstrate that combining physics-informed structure with learned sequence modeling yields meaningful gains for long-horizon PV forecasting. The gap between a physics-only baseline and the integrated configuration indicates that physics alone is insufficient to capture operational PV behavior under weather uncertainty and site-specific effects. Conversely, the improvement of the integrated configuration over a model-only baseline indicates that physics-informed covariates and plausibility shaping contribute beyond what a data-driven model learns implicitly. Taken together, these findings support the broader methodological position that hybrid modeling is well suited to PV forecasting, particularly over long horizons where physical plausibility and robustness are central requirements.

The ablation results further imply that the 30-day forecasting problem is not adequately addressed by a single modeling component. Horizon-isolated variants exhibit substantial degradation, especially the long-horizon-only configuration. This is consistent with the interpretation that near-term and long-term forecasting rely on different error modes and information sources, and that hierarchical reconciliation across temporal resolutions is beneficial.

## 6.2 Data completeness and operational realism

The dataset exhibits heterogeneous completeness across plants and years. Some plant-year exports contain a complete timestamp grid with partially missing PV power values, while others contain extensive or even complete missingness in the measurement field. Such heterogeneity is characteristic of operational industrial data and underscores the importance of robust preprocessing and disciplined evaluation. A key methodological choice in this thesis is to anchor the headline backtest for the target plant to a dedicated ground-truth export for the evaluation year. This choice reduces the risk of evaluating on proxy targets and strengthens the interpretability of reported performance.

Operational realism also includes handling common metering artifacts such as duplicate timestamps and non-uniform base cadence. These artifacts are not treated as exceptional; rather, they motivate deterministic cleaning rules and careful resampling that preserve temporal integrity.

## 6.3 Data leakage prevention and evaluation credibility

The credibility of time-series forecasting results depends critically on preventing data leakage. Leakage can occur not only through direct inclusion of future targets, but also through subtle mechanisms such as fitting normalization statistics on the full dataset, constructing windows that overlap evaluation periods, or selecting hyperparameters with feedback from the test set. In PV forecasting, leakage risks are amplified by seasonal regularity and temporal autocorrelation.

The thesis addresses these risks through strict temporal isolation. For the target test plant (plant_03), the evaluation year 2024 is reserved exclusively for backtesting and is not used in any phase of model fitting, hyperparameter selection, or preprocessing parameter estimation. All transformations that learn parameters from data are fitted on the training period only and then applied unchanged to the evaluation period. This separation produces a clean out-of-sample assessment that mirrors deployment: models trained on historical data are evaluated on a future period.

This design also reduces selection bias. By defining the evaluation year in advance and treating it as a sealed benchmark, the study limits the possibility of tuning decisions that inadvertently optimize for the evaluation period. From an academic standpoint, this strengthens internal validity and supports the claim that the reported metrics reflect genuine generalization rather than optimistic bias.

## 6.4 Reinforcement learning: neutral results and implications

The reinforcement-learning meta-controller yields a near-neutral result under the evaluated configuration. Several factors can lead to neutral outcomes, including insufficiently informative state features, reward definitions that do not fully capture operational objectives, limited exploration, or mismatch between training environments and evaluation regimes. In practical forecasting systems, a controller’s success depends on stable signals that combine error, uncertainty, and operational costs.

The present thesis nonetheless provides a valuable methodological contribution by establishing a controller evaluation framework that is aligned with the same strict backtesting regime used for forecasting performance. This framework makes future improvements scientifically testable. Future work should consider richer state representations, explicit uncertainty modeling, and reward shaping that reflects domain-specific cost functions.

## 6.5 Limitations and generalization

Because plants are anonymized and the headline evaluation focuses on a single target plant for the future-year backtest, care is required when generalizing conclusions. The results support general claims about the value of physics-informed features and temporal embeddings, but external validity would be strengthened by repeating the same leakage-free backtest design across multiple target plants with reliable ground truth.

A further limitation is that the ground-truth evaluation period does not span a full calendar year. While it covers multiple seasons and thus includes substantial variability, future evaluation on complete annual ground truth would strengthen claims about winter and shoulder-season behavior.

## 6.6 Summary

The discussion highlights that MiRACLE’s performance gains are consistent with hybrid modeling theory and that strict temporal isolation strengthens evaluation credibility. The dataset’s operational missingness patterns reinforce the importance of reliable ground truth and disciplined splits. The reinforcement-learning component does not yet yield gains under the evaluated configuration, but the established evaluation framework enables rigorous future refinement.
