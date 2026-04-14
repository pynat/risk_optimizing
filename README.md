# UNDER CONSTRUCTION

# De Prado ML Framework with Causal Discovery: Regime Prediction in ETH Dollar Bars

## Overview

Marcos López de Prado argues that most machine learning models in finance fail due to **spurious features**, statistical relationships that do not reflect the true data-generating process.

This project implements de Prado's ML framework on ETH/USDT, extended with causal discovery as a methodological guardrail, predicting **volatility regimes** for Volatility targeting, Strategy switching, Risk overlays.

The analysis is conducted on **dollar bars** ($500M threshold), which sample observations based on traded value. This reduces heteroskedasticity, improves serial independence of returns, and aligns data with actual market activity.

---

### Research Question

*Can causal feature selection and volatility regime conditioning produce a predictive edge that survives walk-forward validation with a deployable trading strategy?*

### Methodology

- **Dollar Bar Construction** $500M threshold, 2431 bars over 3 years, validated via Durbin-Watson (1.94) and Ljung-Box (lag 10: p=0.65, lag 20: p=0.83)
- **Feature Engineering** volatility, volume, drawdown, technical and order flow features, all ADF-tested for stationarity (36/36 passed)
- **Correlation Analysis** multicollinearity hygiene via clustering (threshold 0.85), not feature selection
- **Causal Discovery** PCMCI map feature dependency structure and identify true drivers vs downstream nodes
- **Triple Barrier Labeling** pt_sl=[2.5, 2.5] ATR, max hold=20 bars, 
  1.4% single-step breach rate, 11.1% timeout rate
- **Random Forest with Purged K-Fold CV** n=5 folds, embargo=1%, prevents temporal leakage
- **MDI/MDA Feature Importance** cross-referenced with causal graph to validate signal vs noise
- **Per-Regime Model Evaluation** separate RF per regime, high-vol identified 
  as sole regime with positive edge (auc=0.658, delta=+0.062 vs global)
- **Walk-Forward Backtest** expanding window, hmm+rf refitted every 50 bars, 
  trade signal active only in predicted high-vol regime, no future leakage


## Results (Out-Of-Sample, Purged CV)

| Metric | Value |
|---|---|
| Regime model accuracy | 0.632, lift +19.5pp over baseline |
| AUC low / med / high | 0.859 / 0.723 / 0.913 |
| Return model AUC (regime-conditioned) | 0.596 |
| Return model AUC (high regime) | 0.658 (+0.062 vs global) |
| Backtest total return | +157.3% vs -27.7% buy & hold |
| Backtest Sharpe (walk-forward) | 1.735 |
| Backtest max drawdown | -22.0% vs -65.0% buy & hold |

**Regime Persistence (P(next \| current)):**

| | →low | →med | →high |
|---|---|---|---|
| low | 0.721 | 0.262 | 0.018 |
| med | 0.169 | 0.672 | 0.160 |
| high | 0.012 | 0.250 | 0.738 |

**Regime Classifier Feature Importance (MDI, Mean Across Folds):**

| Feature | Importance |
|---|---|
| `market_stress` | 0.374 |
| `volatility_7b` | 0.290 |
| `bb_position` | 0.108 |
| `vol_persistence` | 0.100 |



---

## ETH Distribution Characteristics

- **Neutral with negligible positive bias mean return** (0.03%) with median (0.06%) indicates weak bullish drift in the sample period
- **Moderate volatility** (2.21% per bar), more stable than time bars due to event-based sampling
- **Near-symmetric distribution** (skewness -0.02) indicates no strong directional asymmetry
- **Low excess kurtosis** (1.36) suggests extreme events typical for crypto returns

![Return Distribution](results/return_distribution_dollar_bars.png)

### Tail Risk Profile

- **VaR 95%: -3.64%** moderate downside risk per dollar-activity event
- **VaR 99%: -5.45%** rare but meaningful tail losses under high-activity regimes

### Serial Independence Validation

Dollar bars are theoretically more i.i.d. than time bars. This was validated empirically:

- **Durbin-Watson: 1.94** near-perfect, no lag-1 autocorrelation
- **Ljung-Box lag 10: p = 0.65** no autocorrelation across first 10 lags
- **Ljung-Box lag 20: p = 0.83** holds across 20 lags

This confirms the core motivation for dollar bars: returns are statistically independent, satisfying the assumption most ML models require.

![Stationarity](results/autocorrelation_dollar_bars.png)

---

## Feature Engineering

Features are grouped by domain and mapped to distributional properties of the return series:

| Domain | Features | Rationale |
|---|---|---|
| Volatility | `volatility_7b`, `vol_regime`, `vol_momentum`, `vol_expansion` | Activity-conditioned regime signals |
| Risk | `drawdown`, `deep_drawdown`, `var_breach_95/99`, `tail_risk_signal`, `position_size_factor` | Tail risk per unit of traded value |
| Extremes | `extreme_streak`, `upside_momentum` | Non-linear shock detection |
| Technical | `bb_width`, `bb_position`, `atr_normalized`, `vwap_distance`, `rsi` | Latent market state proxies |
| Volume / Flow | `volume_change`, `volume_zscore`, `dollar_vol_z`, `dollar_volume` | Information flow and activity intensity |
| Order Flow | `ofi`, `taker_quote`, `taker_sell_vol`, `trades_change`, `trade_intensity_z` | Aggressive order pressure and trade composition |
| Returns | `ret_raw`, `ret_5` | Short-term momentum signals |
| Composite | `market_stress`, `rolling_vol` | Multi-signal regime indicators |

All 36 features passed stationarity validation. Continuous features were confirmed via ADF test (all p < 0.05). Binary and near-constant features were excluded from ADF.
   
---

## Correlation Analysis

Correlation analysis identifies redundant feature groups and controls multicollinearity.
Feature ranking and final selection are performed post-training via MDI/MDA.

Three clusters were identified at a threshold of 0.85:

- **Cluster 1:** `volume` = total traded volume → includes: market buys (taker buy), market sells (taker sell), `taker_base` = sum of all market buy orders in the base asset (corr 0.98)
- **Cluster 2:** `dollar_volume` = price × volume, measures capital flow in USD, `taker_quote`, `taker_sell_vol`= volume of market sell orders (max corr 0.90)
- **Cluster 3:** `vwap_distance`, `rsi`, `bb_position` (max corr 0.89), all encode normalized price location relative to a reference (VWAP, momentum oscillator, volatility bands), producing high lag=0 redundancy

Due to correlation alaysis one would drop features of this group, but we will keep them for further research.

![Correlation Features](results/correlation.png)  
   
---
## Causal Discovery (PCMCI)

Causal discovery via PCMCI (Tigramite, ParCorr, α=0.05, lags 1–5) was used to map
directional dependencies between features, separating true drivers from downstream
nodes before model training.

**Purpose:** Identify genuine predictive signals vs. construction artifacts. 


**Causal Links To Return Targets:**
- `volume_change → ret_raw` (lag 1, 0.23): strongest real signal, abnormal volume
  growth directly predicts the primary return target
- `bb_width → ret_raw` (lag 1, 0.16): band expansion precedes directional move
- `rsi → ret_5` (lag 1–2): momentum extreme predicts multi-bar return direction

**Cross-Domain Links:**
- `rsi / ret_raw → drawdown` (lag 1, 0.78 / 0.88): momentum extremes and recent
  losses predict cumulative drawdown, drawdown is a sink node
- `atr_normalized / vol_momentum / vol_regime → extreme_streak` (lag 1, 0.24–0.42):
  regime-level vol predicts tail event persistence, extreme_streak is a sink node
- `market_stress → vol_momentum` (lag 1, 0.28): composite stress drives vol
  trajectory, vol_momentum is a neutral node

**Construction Artifacts:**
features computed from the same underlying series:
- `ofi → taker_sell_vol / taker_quote` (0.98): by definition, ofi is the normalized
  difference of those two quantities
- `ret_raw / ret_5 / rsi / bb_position → vwap_distance` (0.86–0.99): all price-location
  measures, high lag=0 redundancy persists into lag=1
- `dollar_vol_z / volume_zscore → dollar_volume` (0.83): z-scores of dollar_volume
- `trade_intensity_z → trades_change` (0.91): both derived from `trades`
- `ret_5 ↔ ret_raw` (0.69–0.91): multi-bar vs. single-bar return overlap
- `volume_change → taker_base / taker_quote / taker_sell_vol` (0.45–0.69):
  volume_change is derived directly from `volume`

**Correlation vs. Causation:**

Correlation analysis flagged `rsi`, `bb_position`, and `vwap_distance` as
near-duplicates (corr > 0.85).   
PCMCI shows`bb_position` with outgoing link to `vwap_distance`.
`rsi`in interaction with `ret_5` and a link to `drawdown`.
`vwap_distance` is shown as a sink node, influenced by several features, `bb_width`, `atr_normalized`, `return_5`.

**Sink Nodes, No Outgoing Edges:**
- `vwap_distance`: caused by rsi / bb_position / ret_raw etc.
- `dollar_volume`: pure sink driven by volume_zscore, dollar_vol_z, volume_change
- `taker_quote`, `taker_sell_vol`: sinks by construction via ofi and volume_change
- `taker_base`: peripheral sink, no return links
- `rolling_vol`: near-pure autocorrelation, no outgoing return links
- `extreme_streak`: sink, receives from vol / atr, no return links
- `drawdown`: driven by rsi / ret_raw, result node, not a predictor
- `volume`: driven by `volume_zscore`, `dollar_vol_z`, `rsi`, `atr_normalized`

**Interpretation:** PCMCI results are used as a structural filter.
Binary features were excluded from PCMCI; only continuous/ordinal features tested.

![PCMCI Algorithm With Strongest Edges](results/pcmci_dag.png)
   
---

## Triple Barrier Labeling & Feature Importance

**Triple Barrier Labeling** (de Prado) replaces naive return-direction labels with structurally sound targets:

- **Upper Barrier:** Profit target at 1.5x ATR
- **Lower Barrier:** Stop-loss at 2.0x ATR
- **Vertical Barrier:** Maximum hold of 20 bars

Split: 1423 train / 590 test bars, 20-bar embargo at the boundary to prevent leakage.
Target balance (train): 0.55, near-balanced, no resampling required.

**MDI vs MDA**

![Feature Importance](results/random_forest.png)

A Random Forest was trained on all stationary features with triple barrier labels as target. Two importance measures were computed and cross-referenced with causal discovery results:

- **MDI (Mean Decrease Impurity):** In-sample, computed from tree structure. Fast but biased toward high-cardinality and correlated features.
- **MDA (Mean Decrease Accuracy):** Out-of-sample permutation importance on held-out data. 


| Feature | MDI Rank | MDA | Verdict |
|---|---|---|---|
| `volume` | 1 | +0.0118 | genuine signal, no direct causal return link in PCMCI |
| `bb_width` | 2 | +0.0033 | genuine signal |
| `taker_base` | 3 | +0.0072 | genuine signal |
| `rsi` | 6 | -0.0124 | noise, causally downstream |
| `vwap_distance` | 8 | -0.0033 | noise, confirmed sink node |

RF1 (all features): train 0.701 / test 0.573. The 12.8pp gap signals overfitting.
22 features were dropped: 20 with negative MDA, 2 combined weak.

`volume` carries the strongest MDA signal (+0.0118) despite lacking a direct causal return link in PCMCI. PCMCI tests direct lagged causal paths. `volume` likely acts as a proxy for latent market activity that PCMCI does not resolve into a single direct edge.

**Final Feature Set (10 features, MDI/MDA validated):**

| Feature | Causal Status |
|---|---|
| `bb_width` | direct return link (ret_raw) |
| `atr_normalized` | driver of extreme_streak |
| `drawdown` | sink in PCMCI, but MDI/MDA confirmed |
| `volume` | sink in PCMCI |
| `taker_base` | peripheral in PCMCI, MDA confirmed |
| `position_size_factor` | neutral node |
| `trade_intensity_z` | routes to artifact sinks |
| `extreme_streak` | sink node, MDA confirmed |
| `vol_regime_change` | binary, MDA confirmed |
| `deep_drawdown` | binary, MDA confirmed |

RF2 (final features): train 0.693 / test 0.573, identical test performance with
less than half the features. The 22 dropped features contributed exclusively to
in-sample overfitting, zero test signal.

Notable conflict between methods: `drawdown`, `extreme_streak`, and `taker_base`
were flagged as PCMCI sink nodes but survived MDA selection. Conversely,
`volume_change` and `rsi` showed direct return links in PCMCI but were eliminated by negative MDA. 

**Purged K-Fold Cross Validation**

Standard k-fold leaks future information at fold boundaries due to rolling feature windows. Purged CV removes training samples whose window overlaps the test period, plus an embargo buffer of 1% of bars (20 bars) after each test fold.

| Fold | Train | Test | Accuracy | AUC |
|---|---|---|---|---|
| 1 | 1607 | 406 | 0.539 | 0.552 |
| 2 | 1587 | 406 | 0.559 | 0.606 |
| 3 | 1587 | 406 | 0.537 | 0.585 |
| 4 | 1587 | 406 | 0.537 | 0.608 |
| 5 | 1604 | 409 | 0.582 | 0.582 |
| **mean** | | | **0.551 ±0.020** | **0.587 ±0.023** |

Naive 70/30 test accuracy was 0.573. Purged CV mean is 0.551, a leakage inflation of +2.2pp, consistent with rolling feature windows bleeding across the split boundary.

AUC of 0.587 on triple barrier labels represents a genuine predictive edge and a
meaningful improvement over the initial full-feature baseline (0.533). Accuracy of 0.551 at a baseline of 0.518 is modest (+3.3pp above chance) but consistent across all five folds with no outlier fold, confirming the edge is structural rather than fold-specific.

---

## Regime Detection & Strategy

Volatility regimes were identified using a Hidden Markov Model (HMM) and labeled
as three discrete states: low (0), medium (1), high (2). The regime classifier
predicts the next bar's regime state using a dedicated 10-feature set.

**Regime Distribution:** low 27.7% / medium 43.8% / high 28.6%.
Majority-class baseline: 43.8%.

**Persistence Structure:**

| Current → Next | Low | Medium | High |
|---|---|---|---|
| Low | 0.721 | 0.262 | 0.018 |
| Medium | 0.169 | 0.672 | 0.160 |
| High | 0.012 | 0.250 | 0.738 |

Regimes are strongly self-persistent (diagonal 0.67–0.74). The classifier
exploits this directly, it predicts whether the current state continues,
not price direction.

**Purged CV Results:**

| Fold | Accuracy | AUC (low) | AUC (med) | AUC (high) |
|---|---|---|---|---|
| 1 | 0.596 | 0.852 | 0.701 | 0.908 |
| 2 | 0.655 | 0.849 | 0.715 | 0.925 |
| 3 | 0.613 | 0.839 | 0.739 | 0.908 |
| 4 | 0.638 | 0.889 | 0.746 | 0.922 |
| 5 | 0.659 | 0.867 | 0.713 | 0.900 |
| **mean** | **0.632** | **0.859 ±0.020** | **0.723 ±0.019** | **0.913 ±0.011** |

Lift over baseline: **+19.5pp**, consistent across all folds. Low and high
regimes are nearly perfectly separable (AUC 0.859 / 0.913). Medium is harder
(0.723), structurally ambiguous as the transition state between extremes.

![Regime Detection](results/regime_detection.png)

Two systematic failure modes are visible: a one-bar lag at transitions, and
systematic underrepresentation of medium regime predictions. Model confidence
rarely exceeds 0.7 except during sustained high-volatility blocks where
`p(high)` briefly spikes above 0.8. `market_stress` and `volatility_7b`
account for 66% of MDI importance, confirming regimes are fundamentally
volatility states.

### Regime-Conditioned Return Prediction:

Two approaches were tested to combine regime and return models:

*Approach 1. Regime Probabilities As Features:*

| Metric | Global | Regime-conditioned | Delta |
|---|---|---|---|
| Accuracy | 0.551 | 0.564 | +0.014 |
| AUC | 0.587 | 0.596 | +0.009 |

Marginal improvement. The global model already captures implicit regime
structure through `vol_regime_change` and `deep_drawdown`.

*Approach 2. Separate Model Per Regime:*

| Regime | Bars | AUC | Delta vs global |
|---|---|---|---|
| Low    | 562 | 0.542 | -0.054 |
| Medium | 889 | 0.553 | -0.043 |
| High   | 581 | 0.658 | +0.062 |

The global AUC of 0.596 reflects mixed signal, concentrated in high-volatility periods and diluted by noise in low and medium regimes.


**Walk-Forward Backtest:**

The regime classifier and return model are combined into a single strategy:
active only in predicted high-volatility regimes. Stop-loss at 1.0x ATR, take-profit at 1.5x ATR. Note: barriers diverge from the triple barrier
label construction (pt_sl=[2.5, 2.5]) Both models are re-fitted every
50 bars on past data only, no future information leaks into any prediction.
HMM Viterbi decoding is limited to the current window to prevent look-ahead
in state assignment.

Signal logic: long when `prob_high > 0.55` and `ret_prob > 0.55`, short
when `ret_prob < 0.45`. Stop-loss at 1.0x ATR, take-profit at 1.5x ATR.

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total return | 157.33% | -27.71% |
| Sharpe ratio | 1.735 | 0.057 |
| Max drawdown | -22.01% | -64.98% |
| Total trades | 94 | — |
| Win rate | 61.7% | — |
| Avg trade return | 1.134% | — |

![Equity Curve](results/backtest_equity.png)


**Limitations:** 
94 trades over 1.5 years, benchmark period (Aug 2024 - Mar 2026) was a bear market for ETH. Hyperparameters were selected on the same data period, no fully independent out-of-sample period exists.    
This backtest is a proof-of-concept, not a validated trading system.
  
---


## Usage
```bash
git clone https://github.com/pynat/causality
pip install -r requirements.txt
jupyter notebook inference_and_causality.ipynb
```