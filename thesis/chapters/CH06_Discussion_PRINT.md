# Chapter 6 — Discussion (Print-Ready Draft)

## 6.1 Interpretation of the main findings

The results demonstrate that integrating physics-informed structure with learned sequence modeling yields meaningful improvements for long-horizon PV forecasting. The performance gap between physics-only and integrated forecasting is large, indicating that purely physical modeling does not capture the full complexity of operational PV generation. Conversely, the improvement of MiRACLE over a forecaster without physics integration indicates that physical priors add value beyond what a data-driven model learns implicitly. Taken together, these findings support the methodological position that hybrid modeling is well suited to PV forecasting, particularly over long horizons where uncertainty grows and physical plausibility is essential.

The ablation results further imply that the 30-day forecasting problem cannot be adequately addressed by a single modeling component. Single-horizon variants exhibit substantial degradation, especially for long-horizon-only configurations. This is consistent with the intuition that near-term and long-term forecasting depend on different information sources and error modes: the near term benefits from high temporal resolution and autoregressive dynamics, while longer horizons require stable structural priors and robust handling of meteorological uncertainty.

## 6.2 Data completeness, operational realism, and limitations

A notable property of the dataset is heterogeneous completeness across plants and years. Some plant-year exports contain a complete timestamp grid with partially missing power values, while others contain extensive or complete missingness in the power value field. This heterogeneity reflects real operational data constraints and reinforces the importance of robust preprocessing, careful split design, and evaluation protocols that rely on reliable ground-truth measurements.

The presence of a dedicated ground-truth export for the test plant enables a particularly strong evaluation design. By separating the evaluation signal from plant-year exports that may contain empty value fields, the thesis avoids an important failure mode: evaluating a model on an incomplete or proxy target series. At the same time, the ground-truth export introduces its own operational artifacts, such as duplicate timestamps and a different base cadence. These artifacts are common in industrial metering systems and are not considered deficiencies; rather, they are part of the practical setting that forecasting systems must handle.

## 6.3 Data leakage, bias, and credibility of evaluation

The credibility of forecasting results depends critically on avoiding data leakage. In time-series forecasting, leakage can occur not only through explicit inclusion of future targets, but also through subtler pathways such as computing normalization parameters on the full dataset, constructing windows that overlap the evaluation period, or selecting hyperparameters based on the test period. These risks are amplified in PV forecasting because seasonal structure and autocorrelation can make it easy for models to benefit from even minor leakage.

The thesis addresses this through a strict temporal isolation strategy for the target plant: the evaluation year is reserved exclusively for backtesting and is not used during model fitting, hyperparameter selection, or preprocessing parameter estimation. The separation is enforced at the dataset construction stage, ensuring that no samples from the evaluation year enter the training pipeline. The result is a clean out-of-sample assessment that mimics deployment: models trained on historical data are evaluated on an unseen future period.

This isolation strategy also mitigates bias that could arise from retrospective selection of evaluation periods. Because the evaluation year is fixed in advance and treated as a sealed benchmark, the reported results are less vulnerable to p-hacking or selection bias. From an academic perspective, this strengthens internal validity and supports the claim that the performance metrics reflect genuine generalization rather than tuning to the test set.

## 6.4 Reinforcement learning: interpretation and future refinement

The reinforcement-learning meta-controller yields a near-neutral result under the evaluated configuration. This should be interpreted cautiously. A neutral policy outcome can arise from several causes, including an insufficiently informative state representation, reward functions that do not reflect the intended operational objective, limited exploration during training, or mismatch between the training environment and the evaluation regime. In practical forecasting systems, the meta-controller’s success depends on capturing the right decision levers and providing stable signals that reflect both error and uncertainty.

The present thesis establishes a reproducible evaluation framework for the controller and demonstrates that the controller can be assessed under the same strict backtesting regime as the forecasting models. This is a valuable methodological contribution even when the policy itself does not yet outperform baseline. Future work should focus on improved state features, explicit uncertainty modeling, and reward shaping aligned with operational cost functions (for example, asymmetric penalties for under-forecasting versus over-forecasting in market contexts).

## 6.5 External validity and generalization

Because plants are anonymized and the evaluation focuses on a single target plant for headline backtesting, care is needed when generalizing conclusions. The results nonetheless support broader claims about hybrid modeling for PV forecasting: physics-informed covariates and constraints are likely to remain beneficial across sites, and learned temporal embeddings provide a systematic mechanism to capture latent site behavior. The methodology is designed to be transferable to additional plants and regions provided that similar data streams are available.

A limitation is that the evaluation period for the ground-truth export does not span a complete calendar year. While it covers multiple seasons and thus provides meaningful variability, future evaluation on full-year ground truth would strengthen claims about winter and shoulder-season performance. Additionally, exploring multiple target plants would strengthen external validity, particularly if plants exhibit different regimes of missingness and operational behavior.

## 6.6 Summary

In summary, the thesis demonstrates that a hybrid forecasting architecture combining physics-informed features and learned temporal representation improves long-horizon PV forecasting performance under a strict out-of-sample backtesting protocol. The experimental design emphasizes data leakage prevention and operational realism, strengthening the credibility of the results. The reinforcement-learning component does not yet provide consistent gains but establishes an extensible framework for future research on adaptive decision-making in forecasting pipelines.
