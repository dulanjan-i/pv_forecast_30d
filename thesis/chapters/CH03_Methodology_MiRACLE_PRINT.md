# Chapter 3 — Methodology: MiRACLE Framework Architecture (Print-Ready Draft)

## 3.1 System overview

MiRACLE (Meta-Intelligent Reinforcement-driven Adaptive Control for Learning-based Ensembles) is an end-to-end photovoltaic (PV) power forecasting framework designed to generate operationally realistic forecasts at 15-minute resolution over a 30-day horizon. The methodological motivation is grounded in the recognition that long-horizon PV forecasting is constrained by compounding uncertainty in meteorological inputs, non-stationarity in site behavior, and the need for physically plausible outputs under changing seasonal and atmospheric regimes. MiRACLE addresses these constraints by integrating three complementary components: a physics-informed feature layer, a learned temporal representation, and an adaptive decision mechanism.

From a systems perspective, MiRACLE can be viewed as a structured forecasting pipeline that takes heterogeneous inputs (historical PV generation, meteorological variables, and site metadata) and transforms them into a feature representation suitable for multi-horizon sequence modeling. The forecasting core is a deep learning forecaster that benefits from both explicit physical priors and compact learned embeddings of recent temporal dynamics. The adaptive component is a reinforcement-learning meta-controller that formalizes operational decisions as actions informed by error signals and uncertainty indicators. Although the reinforcement-learning component is evaluated separately from the core forecasting stack, it is treated as a first-class architectural element because it shapes how the system can respond to distribution shift and operational constraints.

## 3.2 Data provenance and anonymization

The PV generation and associated operational data used in this thesis were provided by Syneco Trading GmbH under a data protection agreement. In accordance with the sensitivity of commercial energy data and contractual obligations, all power plant identifiers were anonymized. Throughout this document, plants are referenced only by plant IDs rather than by site names, geographic identifiers, or any other potentially sensitive descriptors. This anonymization is not merely cosmetic: it ensures that the reporting of results, error characteristics, and temporal patterns cannot be directly mapped back to an identifiable asset, while still allowing rigorous scientific evaluation of forecasting methods.

The anonymization constraint influences methodological choices in two ways. First, it motivates evaluation protocols that rely on time-based splits and out-of-sample backtesting rather than on manual site-level stratification informed by domain knowledge about individual locations. Second, it emphasizes careful reporting practices, where the thesis focuses on reproducible methodology and statistical evidence rather than on site-specific narratives.

## 3.3 Core architectural components

### 3.3.1 Physics-informed feature layer

PV generation is governed by well-understood physical mechanisms: solar geometry determines the maximum available irradiance; atmospheric conditions modulate the partitioning of direct and diffuse components; and system constraints shape achievable power output. Purely data-driven models can implicitly learn portions of this structure, but long-horizon forecasting benefits from explicitly encoding physical priors. MiRACLE therefore incorporates a physics-informed feature layer derived from standard solar position and irradiance modeling. These features act as structured covariates that stabilize learning, reduce the burden on the neural network to infer seasonal geometry from limited samples, and support plausibility constraints such as nighttime output approaching zero.

In addition to serving as covariates, physics-derived signals can be used to impose soft constraints on forecasts. In a practical PV forecasting setting, physically implausible behavior (for example, negative power or large discontinuities during clear-sky periods) reduces operational trust. The methodology therefore treats physics features as a mechanism for both representation and constraint: they guide the forecaster and provide a basis for post-processing steps that enforce physically reasonable shape and bounds.

### 3.3.2 Learned temporal representation (LSTM encoder)

MiRACLE uses a recurrent encoder to learn a compact embedding of recent temporal behavior. The central motivation for an encoder is that PV time series exhibit patterns that are not fully captured by instantaneous weather covariates: operational effects, site-specific response to weather, and temporal persistence in deviations from nominal physics can carry predictive value. Rather than relying on raw autoregressive power values alone, the encoder compresses a window of recent observations into an embedding that captures these latent dynamics.

The encoder is trained under a next-step prediction objective on sliding windows. Formally, let the multivariate feature vector at time t be denoted by x_t ∈ ℝ^F, and let a window of length T be X_t = [x_{t−T}, …, x_{t−1}] ∈ ℝ^{T×F}. The encoder f_θ maps X_t to a prediction ŷ_t for the next step. The final hidden state (or an equivalent pooling of hidden states) serves as an embedding h_t ∈ ℝ^H. This embedding is then provided to the downstream multi-horizon forecaster as an additional input channel, enabling the forecaster to condition on recent latent dynamics in a compact form.

A key methodological point is that the encoder is not used as a standalone long-horizon forecaster. Its purpose is to provide transferable temporal information to a model class better suited for multi-horizon forecasting with covariates. This separation of roles improves interpretability of the architecture and reduces the risk that the encoder must simultaneously learn both representation and long-horizon extrapolation.

### 3.3.3 Multi-horizon forecasting model

Long-horizon PV forecasting requires models that can represent both short-term variability and long-term structure. In MiRACLE, the forecasting core is a multi-horizon sequence model capable of conditioning on historical covariates and known future inputs. The key requirement is that the model can express horizon-dependent behavior: the near term is dominated by short-lived meteorological fluctuations and immediate autoregressive effects, while longer horizons depend more strongly on seasonal and climatological structure.

The methodology treats the forecaster as the primary predictive component whose performance is evaluated under an out-of-sample backtesting protocol. The forecaster consumes (i) historical observations, (ii) physics-derived features, and (iii) the learned embedding from the encoder. This combination is intended to balance inductive bias (from physics) with flexibility (from deep learning) while reducing reliance on any single source of information.

### 3.3.4 Reinforcement-learning meta-controller

Operational forecasting pipelines can be viewed as decision-making systems: choices about routing, blending, or reliance on certain inputs influence outcomes under drift and uncertainty. MiRACLE includes a reinforcement-learning meta-controller that frames selected operational decisions as actions chosen to minimize forecasting error under changing conditions. The controller observes signals derived from recent model performance and data quality and outputs a discrete action. The methodological purpose is to evaluate whether a learned policy can match or improve a fixed baseline decision rule.

In this thesis, the reinforcement-learning component is evaluated as an auxiliary research question rather than as the primary driver of the headline forecasting performance. Nevertheless, it is included in the methodology because it establishes an extensible architecture for future work on adaptation, and because its evaluation requires careful separation of training and testing conditions to avoid optimistic bias.

## 3.4 End-to-end workflow and training phases

The overall workflow consists of three conceptual phases: (i) data preprocessing and quality control, (ii) representation learning for the encoder, and (iii) end-to-end forecasting model training and evaluation. Preprocessing transforms raw time series into a regularized temporal grid, resolves missing and duplicate records, and aligns meteorological covariates with PV measurements. Representation learning then trains the encoder on historical windows to produce embeddings. Finally, the multi-horizon forecaster is trained and evaluated under a backtesting protocol that simulates deployment conditions.

A critical methodological emphasis is the disciplined handling of time-based data splits. PV forecasting is especially prone to leakage if future information is inadvertently incorporated through normalization statistics, feature engineering, or window construction. The methodology therefore treats temporal isolation as a first-order requirement and evaluates models only on data that are strictly out-of-sample with respect to training.

## 3.5 Methodological guarantees against data leakage

The thesis adopts a strict, temporally ordered evaluation design. For the target test plant used in the headline backtesting experiments, all backtesting data belong to a dedicated hold-out period that is not used in any stage of model fitting, hyperparameter selection, or normalization parameter estimation. This includes indirect leakage pathways such as scaling factors, imputation models, and feature standardization: all such transformations are learned on the training period only and then applied unchanged to the hold-out period.

The consequence is that the reported performance represents a genuine out-of-sample assessment that reflects real deployment: the model is trained on historical data and evaluated on an unseen future period. This design also minimizes the risk of selection bias. In particular, the target plant and the hold-out year were selected before model evaluation was performed, and the evaluation period was preserved as a sealed benchmark. The discussion chapter revisits this choice and explains why it strengthens the credibility of the results.
