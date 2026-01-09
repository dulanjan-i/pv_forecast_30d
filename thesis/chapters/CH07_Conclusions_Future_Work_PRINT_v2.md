# Chapter 7 — Conclusions and Future Work (Print-Ready)

## 7.1 Conclusions

This thesis presented MiRACLE, a hybrid PV forecasting framework designed to produce 30-day forecasts at 15-minute resolution under operational constraints. The methodological contribution is the integration of physics-informed covariates with learned temporal representations and multi-horizon forecasting, coupled with an extensible supervisory decision layer. The evaluation design emphasizes strict temporal isolation: the headline backtest is conducted on a future period for the target plant that is completely excluded from training and all preprocessing parameter estimation, supporting credible out-of-sample performance claims.

Empirically, the integrated MiRACLE configuration achieved the strongest overall accuracy among the evaluated baselines under the canonical backtesting protocol. The ablation outcomes indicate that neither physics-only modeling nor model-only forecasting is sufficient to achieve the best performance in this setting. Physics-informed features improve learned forecasting beyond a model-only baseline, while learned components capture variability and site behavior not represented by physics alone. Transfer learning yields modest but consistent gains, supporting the hypothesis that pretrained temporal representations improve robustness under distribution shift.

The reinforcement-learning meta-controller, evaluated under the same strict backtesting regime, was near-neutral relative to a strong baseline decision rule. While it does not yet deliver improvements, its inclusion establishes a testable framework for future research on adaptive operational decision-making.

## 7.2 Implications

From a practical perspective, the results suggest that hybrid approaches are promising for industrial PV forecasting where physical plausibility, horizon-dependent uncertainty, and non-stationarity are central challenges. The strict isolation of backtesting data increases confidence that reported performance is not inflated by leakage or retrospective tuning, which is particularly important in energy contexts where forecasting quality can influence operational and economic decisions.

From a scientific perspective, the thesis supports the view that inductive biases grounded in domain physics can complement deep learning. MiRACLE illustrates a synergy in which physics informs structured covariates and plausibility shaping while learning captures residual behavior and complex interactions.

## 7.3 Future work

Several directions follow from the findings. First, evaluation should be extended to additional target plants with reliable ground truth to strengthen external validity and to study how missingness and measurement completeness influence model training and performance. Second, uncertainty quantification should be incorporated more explicitly through calibrated prediction intervals, enabling more principled operational decisions and more meaningful reinforcement-learning reward definitions.

Third, the reinforcement-learning component should be refined with richer state representations and reward shaping aligned with operational cost functions. Incorporating explicit uncertainty measures, drift indicators, and asymmetric error penalties could yield policies that provide consistent gains over baseline decision rules.

Fourth, data quality modeling represents an important opportunity. Given the observed heterogeneity in measurement completeness, approaches that explicitly model missingness and sensor reliability may improve robustness. Finally, incorporating higher-fidelity physical modeling and additional anonymized system metadata could improve both performance and interpretability, particularly under extreme weather regimes.

## 7.4 Closing remarks

MiRACLE provides a structured framework for long-horizon PV forecasting that emphasizes leakage-free evaluation, operational realism, and methodological clarity. The thesis demonstrates performance gains consistent with hybrid modeling theory and establishes a reproducible foundation for continued research at the intersection of physics-informed modeling, deep learning, and adaptive decision-making.
