# Chapter 3 — Methodology: MiRACLE Framework Architecture (Print-Ready)

## 3.1 System overview

MiRACLE (Meta-Intelligent Reinforcement-driven Adaptive Control for Learning-based Ensembles) is an end-to-end photovoltaic (PV) power forecasting framework designed to generate 30-day forecasts at 15-minute resolution under operational constraints. The methodological motivation is grounded in three realities of long-horizon PV forecasting: (i) uncertainty in meteorological inputs grows quickly with lead time, (ii) PV generation exhibits non-stationarity driven by seasonality and operational effects, and (iii) operational deployment requires physically plausible behavior (for example, near-zero output at night and bounded power during daylight). MiRACLE addresses these constraints through a hybrid architecture that combines physics-informed covariates, a learned temporal representation, and a multi-horizon forecaster. An auxiliary reinforcement-learning (RL) meta-controller is included to study adaptive operational decision-making under drift and uncertainty.

## 3.2 Data provenance and anonymization

The PV generation and associated operational data used in this thesis were provided by Syneco Trading GmbH under a data protection agreement. Because of the sensitivity of commercial energy data and contractual obligations, the power plant names and identifying metadata were anonymized. Throughout this document, plants are therefore referenced only by plant IDs. This anonymization is substantive: it prevents direct attribution of observed generation patterns or forecast errors to a particular identifiable asset while retaining the scientific integrity of the evaluation.

Anonymization also motivates a conservative evaluation strategy. Rather than relying on site-specific narrative interpretation, the thesis emphasizes temporally isolated backtesting and statistical evaluation. This framing supports both privacy compliance and methodological credibility.

## 3.3 Architectural components

MiRACLE is organized as a structured forecasting pipeline. Historical PV generation and meteorological variables are first cleaned and aligned to a regular time grid. Physics-derived covariates are computed from meteorological inputs to encode solar geometry and irradiance structure. In parallel, a recurrent encoder learns a compact representation of recent temporal dynamics from sliding windows of historical observations. These representations are fused with physics-informed covariates and are consumed by a multi-horizon forecasting model based on the Temporal Fusion Transformer (TFT) class. Finally, a hierarchical post-processing step reconciles the multi-resolution outputs and enforces basic physical plausibility. The RL meta-controller is conceptualized as a supervisory layer that can adapt operational decisions based on error and data-quality signals.

## 3.4 Learned temporal representation: LSTM encoder

### 3.4.1 Sliding-window formulation

Let the multivariate feature vector at time index t be denoted by x_t, with F features that may include autoregressive power values and meteorological covariates. A sliding window of length T is constructed from the past T observations, and the encoder is trained to predict the next-step capacity-normalized power value. The window construction is expressed as follows.

$$
\mathbf{X}_{t} = [\mathbf{x}_{t-T}, \ldots, \mathbf{x}_{t-1}] \in \mathbb{R}^{T \times F}
$$

$$
\hat{y}_{t} = f_{\theta}(\mathbf{X}_{t})
$$

The embedding used downstream is the final hidden state of the recurrent network and is denoted by h_t with dimensionality H.

$$
\mathbf{h}_{t} \in \mathbb{R}^{H}
$$

The training objective for the encoder uses a next-step regression loss. In this thesis, the mean squared error formulation is used to convey the objective.

$$
\mathcal{L}_{\mathrm{enc}}(\theta) = \frac{1}{N} \sum_{t=1}^{N} \left( y_{t} - \hat{y}_{t} \right)^{2}
$$

### 3.4.2 Encoder hyperparameter selection (controlled grid sweep)

The initial encoder architecture was selected using a controlled grid sweep on a utility-scale PV dataset used for exploratory pretraining. The sweep varied the hidden size, number of recurrent layers, and the learning rate while holding other factors constant. The searched values were as follows.

$$
H \in \{32, 64, 128\}
$$

$$
L \in \{1, 2\}
$$

$$
\eta \in \{5 \times 10^{-4}, 10^{-3}\}
$$

The sweep held the following parameters fixed: a 24-hour encoder context window corresponding to 96 steps at 15-minute resolution, dropout probability 0.1, batch size 256, and a maximum of 20 epochs.

The canonical encoder configuration selected from this sweep uses hidden size 64, two recurrent layers, dropout 0.1, and learning rate 1×10^-3. This configuration was used as the starting point for transfer and adaptation.

### 3.4.3 Regional adaptation and target-plant fine-tuning

After exploratory pretraining, the encoder is adapted to the regional domain to reduce distribution shift. Regional adaptation is performed under strict no-leak rules: sliding windows are generated within plant boundaries (never across plants), normalization and scaling parameters are computed only on the training portion of each split, and timestamp regularity checks prevent windows from spanning data gaps.

For the target plant used in headline evaluation (plant_03), fine-tuning is performed by initializing from the canonical encoder and applying a conservative learning rate to avoid catastrophic forgetting. The fine-tuning learning rate is reduced by an order of magnitude relative to the sweep default, with a typical value of 1×10^-4. Feature ordering is held fixed across pretraining and fine-tuning to preserve compatibility of learned weights.

## 3.5 Physics-informed covariates and plausibility

PV generation is governed by solar geometry, irradiance partitioning, and system constraints. MiRACLE incorporates physics-informed features derived from standard irradiance decomposition and plane-of-array (POA) transformations. A representative POA decomposition is expressed below, where the POA irradiance is decomposed into direct, sky-diffuse, and ground-reflected components.

$$
G_{\mathrm{POA}} = \mathrm{DNI} \cdot \cos(\theta_{i}) + \mathrm{DHI} \cdot F_{\mathrm{sky}} + \mathrm{GHI} \cdot \rho_{g} \cdot F_{\mathrm{ground}}
$$

Here, theta_i denotes the incidence angle and rho_g denotes the ground albedo. These features serve two roles: they provide structured covariates that reduce the hypothesis space the neural forecaster must learn, and they provide a basis for physically plausible shaping of forecasts (for example, constraining output near night and limiting implausible peaks).

## 3.6 Multi-horizon forecasting: Temporal Fusion Transformers

The forecasting core uses Temporal Fusion Transformers because they are well-suited to multi-horizon sequence prediction with heterogeneous covariates and because they support interpretability analyses such as variable-importance and horizon-dependent relevance. MiRACLE uses two TFT forecasters: a short-head model to capture high-resolution near-term dynamics and a long-head model to provide longer-horizon structure at a coarser temporal resolution.

For both TFT models, the architecture is configured with hidden size 128, two recurrent layers in the sequence-processing component, attention head size 4, and dropout 0.1. The forecasting objective uses quantile regression under a quantile loss with the following quantiles.

$$
\mathcal{Q} = \{0.1, 0.5, 0.9\}
$$

The training setup uses learning rate 1×10^-3, batch size 128, and a patience-based learning-rate reduction schedule with patience 3. The optimizer is Ranger, combining Rectified Adam with Lookahead.

The short-head model uses a 24-hour encoder context corresponding to 96 steps at 15-minute resolution and predicts over the near-term region at the same resolution. The long-head model uses a 7-day encoder context corresponding to 168 hours and produces hourly predictions over a 30-day horizon.

To approximate operational use, forecasts are generated on a rolling schedule with stride 7 days, and each run produces a 30-day horizon at 15-minute resolution after hierarchical reconciliation.

## 3.7 Hierarchical inference and physics-aware blending

MiRACLE produces a unified 15-minute forecast by combining the short-head predictions, the long-head predictions (upsampled to 15-minute resolution), and a physics baseline used as a plausibility prior. The long-head outputs are upsampled by distributing each hourly prediction across the four 15-minute sub-intervals using the within-hour shape implied by the physics baseline.

A generic proportional upsampling rule can be expressed as follows. Let y_l,h denote the long-head prediction for hour h, and let b_{h,k} denote the physics baseline value at the k-th 15-minute sub-interval within hour h. The upsampled long-head prediction at sub-interval k is given by:

$$
\hat{y}^{(l,15m)}_{h,k} = \hat{y}^{(l)}_{h} \cdot \frac{b_{h,k}}{\sum_{j=1}^{4} b_{h,j} + \varepsilon}
$$

MiRACLE then forms an ensemble of the short-head prediction and the upsampled long-head prediction using convex weights.

$$
\hat{y}^{(\mathrm{ML})}_{t} = \alpha_{s} \cdot \hat{y}^{(s)}_{t} + (1 - \alpha_{s}) \cdot \hat{y}^{(l,15m)}_{t}
$$

The ensemble prediction is blended with the physics baseline using another convex combination.

$$
\hat{y}^{(\mathrm{blend})}_{t} = \alpha_{\mathrm{ML}} \cdot \hat{y}^{(\mathrm{ML})}_{t} + (1 - \alpha_{\mathrm{ML}}) \cdot b_{t}
$$

Finally, hard plausibility constraints are applied by clamping to a physically reasonable range and enforcing near-zero output during night-like conditions as indicated by the physics baseline.

## 3.8 Reinforcement-learning meta-controller

The RL meta-controller is formulated as a discrete-action Markov decision process. The state aggregates recent forecasting performance signals (including horizon-conditioned errors), drift indicators, forecast age, and data-quality measures. The action space is discrete and represents operational choices such as routing, blending adjustments, and cadence decisions. The reward penalizes forecasting error and instability while incorporating efficiency costs.

A representative reward form is expressed as follows, where the terms correspond to accuracy, consistency between model components, drift, and an operational cost term.

$$
R_{t} = - w_{\mathrm{acc}} \cdot \mathrm{RMSE}_{t} - w_{\mathrm{cons}} \cdot \Delta_{t} - w_{\mathrm{drift}} \cdot D_{t} - w_{\mathrm{eff}} \cdot C_{t}
$$

The controller uses Double DQN with prioritized experience replay, epsilon-greedy exploration, and soft target-network updates. The default hyperparameters used in this thesis include learning rate 1×10^-4, discount factor 0.95, replay buffer capacity 10,000, dropout 0.4 as a regularizer within the policy network, and weight decay 1×10^-3.

## 3.9 Summary

This chapter presented MiRACLE as a hybrid forecasting architecture that combines physics-informed covariates, a learned temporal encoder, and multi-horizon transformer-based forecasting under a hierarchical inference strategy. The methodological design emphasizes physically plausible outputs and disciplined training–evaluation separation. These choices are essential for producing credible, deployment-relevant results and for supporting rigorous ablation and backtesting analysis in the subsequent chapters.
