# Chapter 7 — Conclusions and Future Work (Print-Ready Draft)

## 7.1 Conclusions

This thesis presented MiRACLE, a hybrid PV forecasting framework designed to produce 30-day forecasts at 15-minute resolution under operational constraints. The central methodological contribution is the integration of physics-informed priors with learned sequence modeling and an extensible decision layer. The evaluation was conducted under a strict temporal backtesting protocol on an unseen future period for a designated test plant, ensuring that headline results reflect genuine out-of-sample generalization.

Empirically, the integrated MiRACLE configuration achieved the best overall accuracy among the evaluated baselines under the canonical backtest. The results demonstrate that neither physics-only modeling nor model-only forecasting is sufficient to reach the best performance in this setting. Physics-informed features and constraints improve learned forecasting beyond a purely data-driven model, while learned components capture variability and site behavior not represented by physics alone.

The thesis also showed that warm-start transfer learning yields modest but consistent improvements over cold-start variants, supporting the hypothesis that learned temporal representations can provide stability and robustness under distribution shift. Finally, the reinforcement-learning meta-controller was evaluated in a reproducible manner and was found to be near-neutral relative to baseline. While it does not yet deliver gains, the framework establishes a foundation for future research on adaptive decision-making in forecasting pipelines.

## 7.2 Implications

From a practical perspective, the results indicate that hybrid methods are promising for industrial PV forecasting where physical plausibility, uncertainty growth over horizon, and non-stationarity are central challenges. The strict isolation of backtesting data strengthens confidence that reported performance is not inflated by leakage or retrospective tuning. This is especially important in energy applications, where performance claims often influence operational decisions and economic outcomes.

From a scientific perspective, the thesis supports the view that inductive biases grounded in domain physics can complement deep learning. Rather than treating physics and learning as competing paradigms, MiRACLE demonstrates their synergy: physics informs structure and constraints, while learning captures residual behavior and complex interactions.

## 7.3 Future work

Several directions are suggested by the findings. First, evaluation should be extended to additional target plants to strengthen external validity and to study how data completeness patterns affect model training and performance. Second, uncertainty quantification should be incorporated more explicitly. Long-horizon forecasting is fundamentally uncertain, and calibrated prediction intervals would support decision-making and enable more principled reinforcement-learning rewards.

Third, the reinforcement-learning component should be refined with improved state representations and reward shaping aligned with operational cost functions. In particular, incorporating uncertainty estimates, explicit drift indicators, and market-relevant penalties could yield policies that improve over baseline decision rules.

Fourth, preprocessing and data quality modeling represent an important research opportunity. The heterogeneity in measurement completeness suggests that explicitly modeling missingness and sensor reliability could improve robustness. Methods such as probabilistic imputation, missingness-aware architectures, or joint models of measurement and generation could be explored.

Finally, integrating higher-fidelity physical modeling and site metadata could improve both accuracy and interpretability. While physics-derived covariates provide substantial benefits, additional information about system configuration and maintenance events could further reduce systematic bias and improve long-horizon behavior.

## 7.4 Closing remarks

MiRACLE provides a structured framework for long-horizon PV forecasting that emphasizes operational realism, leakage-free evaluation, and methodological clarity. The thesis demonstrates meaningful performance gains from hybrid modeling and establishes a reproducible foundation for continued research at the intersection of physics-informed modeling, deep learning, and adaptive decision-making.
