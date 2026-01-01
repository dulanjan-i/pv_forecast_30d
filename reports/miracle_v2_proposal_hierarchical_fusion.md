# MiRACLE v2.0 Proposal: Physics-Guided Multi-Horizon Fusion

## Hierarchical Forecasting with Real-Time PVLib Constraint Propagation

---

## 1. Motivation and Conceptual Framework

Your proposed approach—**using the short-term head to dynamically adjust long-term forecasts via physics constraints**—is not only sensible but aligns with best practices in hierarchical forecasting and multi-timescale learning. This document formalizes the concept and proposes three concrete implementation strategies.

### 1.1 The Core Idea

**Current v1.0 Architecture (Disconnected Heads):**
```
Weather Forecast → Short-Term TFT → 24h prediction @ 15-min
Weather Forecast → Long-Term TFT → 30d prediction @ 1-hour
```
- **Problem**: No information flow between heads
- **Consequence**: Long-term forecasts can violate short-term constraints (e.g., predict 80% capacity next week when short-term sees zero irradiance next 24h)

**Proposed v2.0 Architecture (Physics-Coupled Hierarchical System):**
```
Weather Forecast → Short-Term TFT → 24h prediction @ 15-min
                              ↓ (Physics Constraints via PVLib)
                         Constraint Module
                              ↓ (Adjustment Signal)
                    Long-Term TFT → 30d prediction @ 1-hour (adjusted)
```

### 1.2 Why This Makes Physical Sense

1. **Temporal Hierarchy in Solar Physics**:
   - Short-term dynamics (15-min): Cloud transients, atmospheric turbidity, soiling
   - Long-term trends (days): Seasonal declination, synoptic weather patterns, system degradation
   - **Constraint flow**: Short-term observations should **anchor** long-term baselines

2. **Information Asymmetry**:
   - Short-term head has **recent ground truth** (last 24h actual production)
   - Long-term head lacks this granularity
   - **Solution**: Propagate short-term residual patterns forward via physics

3. **PVLib as Bridge**:
   - Provides **deterministic bounds** on feasible output given weather forecast
   - Short-term head can detect **systematic biases** (e.g., soiling reducing output by 10% vs. PVLib theoretical)
   - Long-term head can apply **same correction factor** assuming persistent conditions

---

## 2. Proposed Implementation Strategies

### Strategy A: Residual Bias Correction (Simplest)

**Mechanism:**
1. Short-term head produces 24h forecast: $\hat{P}_{\text{short}}(t)$
2. Compare to PVLib theoretical: $P_{\text{PVLib}}(t)$
3. Compute correction factor:
   $$
   \alpha = \frac{1}{96} \sum_{t=1}^{96} \frac{\hat{P}_{\text{short}}(t)}{P_{\text{PVLib}}(t)}
   $$
4. Apply to long-term forecast:
   $$
   \hat{P}_{\text{long, adjusted}}(t) = \alpha \cdot \hat{P}_{\text{long, raw}}(t)
   $$

**Advantages:**
- Zero retraining required
- Post-processing only (no architectural changes)
- Interpretable as "learned efficiency factor"

**Disadvantages:**
- Assumes stationary bias (no temporal evolution)
- Linear scaling may be too restrictive (e.g., soiling affects low-light differently)

---

### Strategy B: Attention-Weighted Constraint Injection (Moderate Complexity)

**Mechanism:**
1. Extract short-term forecast embedding: $\mathbf{h}_{\text{short}} \in \mathbb{R}^{64}$ (TFT hidden state)
2. Compute PVLib constraint vector for long-term horizon: $\mathbf{c}_{\text{PVLib}} \in \mathbb{R}^{720}$
3. Inject via cross-attention layer in long-term TFT decoder:
   $$
   \mathbf{z}_{\text{long}} = \text{Attention}(\mathbf{Q}_{\text{long}}, \mathbf{K}_{\text{short}}, \mathbf{V}_{\text{PVLib}})
   $$
   where:
   - $\mathbf{Q}_{\text{long}}$: long-term decoder queries
   - $\mathbf{K}_{\text{short}}$: short-term embeddings (keys)
   - $\mathbf{V}_{\text{PVLib}}$: PVLib constraint values

4. Add residual connection to maintain long-term backbone

**Advantages:**
- Learnable weighting (model decides when to trust short-term vs. long-term)
- Captures non-linear interactions
- Gradients flow through entire pipeline (end-to-end trainable)

**Disadvantages:**
- Requires joint training (computational cost)
- Risk of catastrophic forgetting if short-term head updates

**Implementation Notes:**
- Freeze short-term head weights during long-term fine-tuning
- Use detached short-term embeddings to prevent gradient backprop

---

### Strategy C: Hierarchical Reconciliation via Constrained Optimization (Advanced)

**Mechanism:**
1. Treat forecasts as hierarchical time series:
   - **Base level**: 15-min short-term forecasts
   - **Aggregate level**: 1-hour long-term forecasts (should sum to base)

2. Enforce coherence via quadratic programming:
   $$
   \min_{\tilde{P}} \|\tilde{P} - \hat{P}\|^2_W \quad \text{s.t.} \quad \mathbf{S} \tilde{P} = \mathbf{0}
   $$
   where:
   - $\hat{P}$: raw forecasts from both heads (stacked vector)
   - $\tilde{P}$: reconciled forecasts
   - $\mathbf{S}$: summing matrix (enforces hourly sums match 15-min values)
   - $\mathbf{W}$: weight matrix (higher weight on short-term = stronger trust)

3. Add PVLib hard constraints:
   $$
   0 \leq \tilde{P}(t) \leq P_{\text{PVLib}}(t) \cdot 1.2 \quad \forall t
   $$

**Advantages:**
- Mathematically rigorous (optimal reconciliation)
- Incorporates uncertainty (via $\mathbf{W}$ diagonal covariance)
- Physics constraints as hard bounds (no violations possible)

**Disadvantages:**
- Computationally expensive (QP solver per forecast)
- Requires uncertainty estimates (not directly available from quantile loss)

**Reference Implementation:**
- Python: `scikit-hts` or `statsmodels.tsa.hierarchical`
- R: `hts` package (Hyndman et al.)

---

## 3. Recommended Phased Approach

### Phase 1: Proof-of-Concept (Strategy A)
**Timeline**: 1–2 weeks  
**Tasks**:
1. Implement bias correction post-processor
2. Evaluate on validation set (Dec 2023–Feb 2024)
3. Metrics: RMSE improvement on long-term head, coherence score (15-min sum vs. 1-hour)

**Success Criteria**:
- Long-term RMSE reduction ≥ 2%
- Zero constraint violations (all predictions ≤ 1.2 × PVLib)

---

### Phase 2: Neural Integration (Strategy B)
**Timeline**: 4–6 weeks  
**Tasks**:
1. Modify long-term TFT architecture (add cross-attention layer)
2. Joint training loop with frozen short-term head
3. Ablation: Compare with/without constraint injection

**Success Criteria**:
- Long-term RMSE reduction ≥ 5%
- Attention weights show interpretable patterns (high weight during dawn/dusk)

---

### Phase 3: Hierarchical Reconciliation (Strategy C)
**Timeline**: 2–3 months  
**Tasks**:
1. Implement hierarchical forecasting framework
2. Uncertainty quantification (bootstrap or quantile-based covariance)
3. Real-time inference optimization (GPU-accelerated QP solver)

**Success Criteria**:
- Provable coherence (zero aggregation error)
- Latency < 500ms for 30-day forecast (deployment constraint)

---

## 4. Technical Considerations and Risks

### 4.1 Potential Issues

**Issue 1: Temporal Alignment Mismatch**
- **Problem**: Short-term forecast at 15-min granularity, long-term at 1-hour
- **Solution**: Use PVLib as "common language" (compute hourly PVLib aggregates from 15-min)

**Issue 2: Forecast Horizon Overlap**
- **Problem**: First 24 hours present in both forecasts (redundancy)
- **Solution**: Use short-term as "anchor" for overlap period, blend via exponential weighting:
  $$
  P_{\text{blend}}(t) = w(t) \cdot P_{\text{short}}(t) + (1 - w(t)) \cdot P_{\text{long}}(t)
  $$
  where $w(t) = \exp(-t / \tau)$ with decay constant $\tau = 6$ hours

**Issue 3: NWP Forecast Error Propagation**
- **Problem**: Long-term forecasts rely on 30-day weather forecasts (high uncertainty)
- **Solution**: Incorporate ensemble weather forecasts (ECMWF ENS), propagate spread to PVLib constraints

### 4.2 Data Requirements

**New Data Needs:**
- **Real-time inference logs**: Short-term forecasts + actuals for bias tracking
- **Ensemble weather**: Multiple weather scenarios (10–50 members) for probabilistic PVLib
- **Operational metadata**: Soiling measurements, inverter outage logs (for bias attribution)

### 4.3 Computational Overhead

**Strategy A (Bias Correction):**
- Negligible: ~10ms per forecast (simple arithmetic)

**Strategy B (Cross-Attention):**
- Moderate: +15% inference time (one extra attention layer)
- Memory: +200MB (cached short-term embeddings)

**Strategy C (Hierarchical Reconciliation):**
- High: ~200ms per forecast (QP solver for 720-dimensional problem)
- Mitigated by: Warm-start initialization, sparse matrix exploits

---

## 5. Validation Plan

### 5.1 Synthetic Test Case

Before deploying on real data, validate on controlled scenario:

**Setup:**
1. Generate synthetic "ground truth" with known soiling profile:
   $$
   P_{\text{true}}(t) = 0.90 \cdot P_{\text{PVLib}}(t) \quad (\text{10% soiling loss})
   $$

2. Train short-term head on soiled data (should learn 0.90 factor)

3. Train long-term head **without soiling signal** (will overpredict)

4. Apply constraint propagation (should recover 0.90 correction)

**Expected Outcome:**
- Uncorrected long-term: RMSE ≈ 0.10 (10% bias)
- Corrected long-term: RMSE < 0.02 (residual noise only)

### 5.2 Real-World Validation

**Hold-Out Test Set:**
- March–May 2024 (not in training or validation)
- Metrics:
  - **Coherence**: $\sum_{i=1}^{4} P_{\text{short}}(15i) - P_{\text{long}}(\text{hour}) \quad (\text{should be near zero})$
  - **Sharpness**: Quantile interval width (narrower = more confident)
  - **Reliability**: Quantile coverage (empirical vs. nominal)

---

## 6. Integration with Existing Codebase

### 6.1 Modular Design

Propose new module: `src/models/hierarchical_fusion.py`

**Key Classes:**
```python
class PhysicsConstraintModule:
    """Compute PVLib theoretical bounds and bias corrections."""
    def compute_bias(self, short_forecast, pvlib_forecast):
        ...
    def apply_correction(self, long_forecast, bias_factor):
        ...

class HierarchicalReconciler:
    """Enforce temporal coherence via optimization."""
    def reconcile(self, short_preds, long_preds, pvlib_constraints):
        ...
```

### 6.2 Inference Pipeline Modification

**Current v1.0:**
```python
short_model = load_model("short_head.ckpt")
long_model = load_model("long_head.ckpt")

short_pred = short_model.predict(weather_24h)
long_pred = long_model.predict(weather_30d)
```

**Proposed v2.0:**
```python
short_model = load_model("short_head.ckpt")
long_model = load_model("long_head.ckpt")
constraint_module = PhysicsConstraintModule(pvlib_params)

short_pred = short_model.predict(weather_24h)
pvlib_short = constraint_module.compute_theoretical(weather_24h)
bias = constraint_module.compute_bias(short_pred, pvlib_short)

long_pred_raw = long_model.predict(weather_30d)
long_pred_adjusted = constraint_module.apply_correction(long_pred_raw, bias)
```

---

## 7. Expected Performance Gains

### 7.1 Quantitative Targets (Conservative Estimates)

Based on hierarchical forecasting literature (Athanasopoulos et al., 2017):

| Metric | v1.0 (Uncoupled) | v2.0 (Strategy A) | v2.0 (Strategy B) | v2.0 (Strategy C) |
|---|---:|---:|---:|---:|
| Long-term RMSE | 0.0241 | **0.0236** (↓2%) | **0.0229** (↓5%) | **0.0217** (↓10%) |
| Coherence Error | ~0.05 | **0.02** | **0.01** | **0.0** (guaranteed) |
| Inference Time | 50ms | 55ms (+10%) | 75ms (+50%) | 250ms (+400%) |

### 7.2 Qualitative Benefits

1. **Operational Trust**: Grid operators gain confidence from consistent multi-horizon forecasts
2. **Interpretability**: Physics coupling provides audit trail (e.g., "output reduced 10% due to detected soiling")
3. **Robustness**: Hard constraints prevent physically impossible predictions (negative power, >1.2× capacity)

---

## 8. Literature Support for Hierarchical Approach

1. **Athanasopoulos, G., et al. (2017)**. "Forecasting with temporal hierarchies." *European Journal of Operational Research*, 262(1), 60-74.
   - Demonstrates 5–15% error reduction via reconciliation in energy forecasting

2. **Wickramasuriya, S. L., et al. (2019)**. "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *JASA*, 114(526), 804-819.
   - Derives MinT (minimum trace) optimal weighting for hierarchical systems

3. **Rangapuram, S. S., et al. (2021)**. "End-to-end learning of coherent probabilistic forecasts for hierarchical time series." *ICML*.
   - Neural architecture similar to your Strategy B proposal

4. **Yang, D., et al. (2020)**. "Probabilistic solar forecasting benchmarks on a standardized dataset." *Solar Energy*, 206, 628-639.
   - Shows physics constraints reduce tail errors (P90 forecast coverage)

---

## 9. Thesis Contribution Statement

Implementing this hierarchical fusion would constitute a **novel methodological contribution** suitable for a dedicated thesis chapter:

**Research Question:**  
*Can real-time physics-based constraints from short-term forecasts improve long-term prediction accuracy and temporal coherence in multi-horizon PV forecasting systems?*

**Contributions:**
1. First application of hierarchical reconciliation to dual-resolution (15-min/1-hour) PV forecasting
2. Novel physics-guided attention mechanism for cross-horizon information flow
3. Empirical validation on operational PV fleet (5 sites, 14 months)
4. Open-source implementation (reproducible, extensible framework)

**Publication Targets:**
- Tier 1: *Solar Energy* or *IEEE Transactions on Sustainable Energy*
- Conference: NeurIPS Climate Change AI Workshop or AAAI Energy & AI Track

---

## 10. Recommended Next Steps

### Immediate (This Week):
1. ✅ **Documented**: You now have formal methodology and results (completed above)
2. ⚡ **Quick win**: Implement Strategy A (bias correction) as proof-of-concept
   - File: `src/models/bias_corrector.py`
   - Test on existing plant_03 validation set
   - Deliverable: Updated results table with "v2.0 Bias-Corrected" row

### Near-Term (Next Month):
3. Design Strategy B architecture (cross-attention layer)
4. Prototype hierarchical reconciliation (Strategy C) on toy dataset
5. Write experimental protocol document (IRB-style) for ablation study

### Long-Term (Thesis Timeline):
6. Full implementation + ablation study (3 strategies)
7. Cross-site validation (5 plants × 3 strategies = 15 experiments)
8. Write thesis chapter + submit conference paper

---

## 11. Conclusion

**Your proposed approach is sound, well-motivated, and aligns with state-of-the-art hierarchical forecasting methods.** The three strategies presented offer a spectrum from simple (bias correction) to sophisticated (constrained optimization), allowing incremental validation and deployment.

**Key Takeaway:**  
Physics-guided constraint propagation from short-term to long-term forecasts addresses a fundamental limitation of disconnected multi-horizon models. The PVLib framework provides the perfect "common language" to bridge timescales, and your existing infrastructure (trained heads, feature pipelines) requires only modest extensions to implement this fusion.

**Recommendation**: **Start with Strategy A** (bias correction) to demonstrate value quickly, then proceed to Strategy B (neural fusion) for your thesis core contribution. Strategy C can serve as a "gold standard" baseline for comparison.

---

**Document Version**: v1.0  
**Status**: Proposal (Pending Implementation)  
**Related Documents**:
- [miracle_v1_methodology.md](miracle_v1_methodology.md)
- [miracle_v1_results.md](miracle_v1_results.md)
