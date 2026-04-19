# De Prado ML Framework with Causal Discovery: Regime Prediction + Strategy for ETH Dollar Bars

## Overview

Marcos López de Prado argues that most machine learning models in finance fail due to **spurious features**, statistical relationships that do not reflect the true data-generating process.

This project implements de Prado's ML framework on ETH/USDT, extended with causal discovery as a methodological guardrail, predicting **volatility regimes** for Volatility targeting, Strategy switching, Risk overlays.

The analysis is conducted on **dollar bars** ($500M threshold), which sample observations based on traded value. This reduces heteroskedasticity, improves serial independence of returns, and aligns data with actual market activity.

---

### Research Question

*Does replacing correlation-based feature selection with causal discovery (PCMCI) improve out-of-sample predictive stability in a regime-conditioned trading system on ETH dollar bars?*

### Methodology

- **Dollar Bar Construction** $500M threshold, 2431 bars over 3 years, validated via Durbin-Watson (1.94) and Ljung-Box (lag 10: p=0.65, lag 20: p=0.83)
- **Feature Engineering** volatility, volume, drawdown, technical and order flow features, all ADF-tested for stationarity (36/36 passed)
- **Correlation Analysis** multicollinearity hygiene via clustering (threshold 0.85), not feature selection
- **Causal Discovery** PCMCI map feature dependency structure and identify true drivers vs downstream nodes
- **Triple Barrier Labeling** pt_sl=[2.5, 2.5] ATR, max hold=20 bars, 1.4% single-step breach rate, 11.1% timeout rate
- **Random Forest with Purged K-Fold CV** n=5 folds, embargo=1%, prevents temporal leakage
- **MDI/MDA Feature Importance** cross-referenced with causal graph to validate signal vs noise
- **Per-Regime Model Evaluation** separate RF per regime, high-vol identified as sole regime with positive edge (auc=0.658, delta=+0.062 vs global)
- **Walk-Forward Backtest** expanding window, hmm+rf refitted every 50 bars, trade signal active only in predicted high-vol regime. Leakage prevention: HMM refit at each step on past bars only, Viterbi decoding is applied strictly up to the current bar, all scalers and models are trained exclusively on the expanding window available at prediction time.


## Results (Out-Of-Sample, Purged CV)

| Metric | Value |
|---|---|
| Regime model accuracy | 0.632, lift +19.5pp over baseline |
| AUC low / med / high | 0.859 / 0.723 / 0.913 |
| Return model AUC (regime-conditioned) | 0.596 |
| Return model AUC (high regime) | 0.658 (+0.062 vs global) |
| Backtest total return | 202.63% vs -27.7% buy & hold |
| Backtest Sharpe (walk-forward) | 2.052 |
| Backtest max drawdown | -27.53% vs -65.0% buy & hold |

**Regime Persistence (P(next \| current)):**

| | →low | →med | →high |
|---|---|---|---|
| low | 0.721 | 0.262 | 0.018 |
| med | 0.169 | 0.672 | 0.160 |
| high | 0.012 | 0.250 | 0.738 |




---

## ETH Distribution Characteristics

- **Neutral with negligible positive bias mean return** (0.03%) with median (0.06%) indicates weak bullish drift in the sample period
- **Moderate volatility** (2.21% per bar), more stable than time bars due to event-based sampling
- **Near-symmetric distribution** (skewness -0.02) indicates no strong directional asymmetry
- **Low excess kurtosis** (1.36) lower than typical crypto time-bar distributions (often >5), consistent with dollar bar sampling reducing heteroskedasticity and compressing tail events

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

Causal discovery via PCMCI (Tigramite, ParCorr, α=0.05, lags 1–5) maps
directional time-lagged dependencies between features. Binary features excluded.

**Structural drivers (hub nodes):**
- `ret_raw` drives drawdown (lag1–5, up to 0.57), vwap_distance (0.87), taker_sell_vol (0.40)
- `ofi` drives volume (0.43), taker_sell_vol (0.86), taker_quote (0.75)
- `trade_intensity_z` drives trades_change (0.86), taker_base (0.31)
- `volume_change` drives volume (0.48), taker_base (0.20), taker_sell_vol (0.27)

**Sink nodes (result nodes, no predictive outgoing edges):**
`vwap_distance`, `dollar_volume`, `taker_quote`, `taker_sell_vol`,
`taker_base`, `rolling_vol`, `extreme_streak`, `drawdown`, `volume`

**Construction artifacts (high strength by definition, not signal):**
- `ofi → taker_sell_vol / taker_quote` (0.86–0.98)
- `ret_raw / rsi / bb_position → vwap_distance` (0.85–0.93)
- `dollar_vol_z / volume_zscore → dollar_volume` (0.89)
- `trade_intensity_z → trades_change` (0.86)

### Lag Feature Engineering

PCMCI revealed that hub nodes unfold their causal effect over multiple lags,
which the random forest cannot detect automatically. Based on the strongest
causal links to top RF features (volume, drawdown, taker_base), the following
lag features were engineered explicitly:

```python
df['ret_raw_lag1']           = df['ret_raw'].shift(1)          # → drawdown (0.35), vwap_distance (0.87)
df['ret_raw_lag2']           = df['ret_raw'].shift(2)          # → drawdown (0.57), strongest single link
df['ofi_lag1']               = df['ofi'].shift(1)             # → volume (0.43), taker_sell_vol (0.86)
df['volume_change_lag1']     = df['volume_change'].shift(1)   # → volume (0.48)
df['volume_change_lag2']     = df['volume_change'].shift(2)   # → volume (0.39)
df['trade_intensity_z_lag2'] = df['trade_intensity_z'].shift(2) # → taker_base (0.31)
```


Adding these features to the walk-forward model improved total return and Sharpe ratio.
No leakage: all lags use strictly past values. Minor feature-selection bias exists
as PCMCI was run on the full dataset, but does not see labels.


![PCMCI Algorithm With Strongest Edges](results/pcmci_dag.png)
   
---

## Triple Barrier Labeling & Feature Importance

**Triple Barrier Labeling** (de Prado) replaces naive return-direction labels with structurally sound targets:

For each bar `t`, three barriers are defined. The price path `[t+1, t+max_hold]` is scanned and the first barrier touched determines the label.
 
```
upper    = price_t * (1 + pt_mult * vol_t)   →  label +1  (profit target)
lower    = price_t * (1 - sl_mult * vol_t)   →  label -1  (stop loss)
vertical at t + max_hold                      →  label  0  (timeout)
```
 
`vol_t` is the rolling standard deviation of log returns over a 20-bar lookback window.
Barriers scale with local volatility so thresholds are consistent across high- and low-vol regimes.
 
---
 
## Calibration
 
Before labeling, single-bar breach rates are printed to validate that multipliers are not trivially crossed:
 
```
> 2.5x vol (pt): 1.4% of bars breach in single step
> 2.5x vol (sl): 1.4% of bars breach in single step
```
 
Label distribution is also checked across low / mid / high vol regimes. A spread > 5pp in the profit label
across regimes signals systematic directional bias.
 
---
 
## Label Config
 
| Parameter  | Value |
|------------|-------|
| `pt_mult`  | 2.5   |
| `sl_mult`  | 2.5   |
| `max_hold` | 20    |
 
Timeout bars (`label == 0`, 11.1%) are dropped. Remaining labels are remapped to binary:
`-1 → 0` (stop loss), `1 → 1` (profit target).
 
```
total bars after drop : 2033
class balance         : 0.518  (near-balanced, no resampling required)
```



**MDI vs MDA**

![Feature Importance](results/random_forest.png)

A Random Forest was trained on all stationary features with triple barrier labels as target. Two importance measures were computed and cross-referenced with causal discovery results:

- **MDI (Mean Decrease Impurity):** In-sample, computed from tree structure. Fast but biased toward high-cardinality and correlated features.
- **MDA (Mean Decrease Accuracy):** Out-of-sample permutation importance on held-out data. 


RF1 (all features): train 0.683 / test 0.573. The gap signals overfitting.
Features with negative MDA or in the bottom 30% of combined rank are dropped (24 features removed).

### PCMCI Lag Features in RF

| Feature | MDI rank | MDA | Causal link |
|---|---|---|---|
| `ret_raw_lag1` | mid | positive | → drawdown (0.35), vwap_distance (0.87) |
| `ret_raw_lag2` | mid | positive | → drawdown (0.57), strongest single link |
| `ofi_lag1` | mid | slightly positive | → volume (0.43) |
| `volume_change_lag1` | low | near zero | → volume (0.48) |
| `trade_intensity_z_lag2` | low | near zero | → taker_base (0.31) |

### Causal vs. RF Alignment

| Feature | Causal Status |
|---|---|
| `bb_width` | direct return link (ret_raw) |
| `atr_normalized` | driver of extreme_streak |
| `drawdown` | sink in PCMCI, MDI/MDA confirmed |
| `volume` | sink in PCMCI, strongest MDI |
| `taker_base` | peripheral in PCMCI, MDA confirmed |
| `deep_drawdown` | binary, top MDA |
| `trade_intensity_z` | hub in PCMCI, neutral MDA |

Notable conflicts: `rsi` has direct causal links in PCMCI but strongly negative MDA.
`vol_momentum` and `taker_sell_vol` are negative in MDA despite causal presence.
Both are dropped. `volume` remains the strongest MDI feature despite being a PCMCI
sink, likely proxying latent market activity not resolved into a single causal edge.


RF2 (final features): train 0.683 / test 0.573, identical test performance with less than half the features. The 24 dropped features contributed exclusively to in-sample overfitting, zero test signal.

Notable conflict between methods: `drawdown`, `extreme_streak`, and `taker_base` were flagged as PCMCI sink nodes but survived MDA selection. Conversely, `volume_change` and `rsi` showed direct return links in PCMCI but were eliminated by negative MDA. 



### What PCMCI Added Beyond RF

The RF alone assigned `ofi` low importance because it sees the feature at t=0,
where stronger in-sample splits dominate. PCMCI identified `ofi` as a structural
hub driving `volume` (0.43), `taker_sell_vol` (0.86), and `taker_quote` (0.75)
across lag 1. Explicitly engineering `ofi_lag1` produced the only PCMCI-derived
feature that survived final MDA selection into the 11-feature model. The remaining
lag features (`ret_raw_lag1/2`, `volume_change_lag1/2`, `trade_intensity_z_lag2`)
improved walk-forward performance but were eliminated by MDA as insufficiently
stable out-of-sample. `ofi_lag1` is the single concrete contribution of causal
discovery to the final feature set.


**Purged K-Fold Cross Validation**

Standard k-fold leaks future information at fold boundaries due to rolling feature windows. Purged CV removes training samples whose window overlaps the test period, plus an embargo buffer of 1% of bars (20 bars) after each test fold.

```
folds        : 5
embargo_pct  : 0.0098  (20 bars)
```
 
| Metric    | Mean   | Std    |
|-----------|--------|--------|
| Accuracy  | 0.5548 | 0.0291 |
| AUC       | 0.6096 | 0.0341 |
| Log Loss  | 0.6918 | 0.0239 |
 
```
naive 70/30 test accuracy : 0.5746
purged cv mean accuracy   : 0.5548
leakage inflation estimate: +0.0198
```


---

## Regime Detection & Strategy

A second classifier predicts the **volatility regime of the next bar** (0=low, 1=med, 2=high)
using 10 regime-specific features. 
 
### Features
 
Features are drawn from MDA-validated inputs of the main model, plus regime-specific additions:
`bb_position`, `extreme_streak`, `atr_normalized`, `vol_persistence`, `tail_risk_signal`,
`bb_width`, `volatility_7b`, `volume_zscore`, `dollar_vol_z`, `market_stress`
 
### Model
 
Same `RandomForestClassifier` setup as the main model, `max_depth=5`, `min_samples_leaf=15`.
Purged K-Fold (5 splits) with identical embargo. OOS probabilities are NaN-initialized so
purged/embargoed bars are never silently assigned to class 0.
 
### Results
 
```
baseline accuracy (majority class) : 0.4375
model accuracy (purged cv mean)    : 0.6324
lift over baseline                 : +0.1949
```
 
| Class | AUC mean | AUC std |
|-------|----------|---------|
| low   | 0.8592   | 0.0195  |
| med   | 0.7230   | 0.0192  |
| high  | 0.9126   | 0.0107  |
 
High and low regimes are predicted with strong confidence. Medium regime is harder to separate, as expected given its transitional nature.
 
### Regime Transition Matrix
 
Empirical one-step transition probabilities (row = current, col = next):
 
```
        →low   →med  →high
low    0.721  0.262  0.018
med    0.169  0.672  0.160
high   0.012  0.250  0.738
```
 
Regimes are strongly persistent. Direct low→high transitions are near-zero (1.8%),
which the model implicitly exploits.
 
### Feature Importance For Regime Detection (MDI, mean across folds)
 
```
market_stress    0.3735
volatility_7b    0.2901
bb_position      0.1075
vol_persistence  0.1000
```
 
`market_stress` and `volatility_7b` together account for ~66% of impurity reduction.


![Regime Detection](results/regime_detection.png)



### Regime-Conditioned Return Prediction:
 
Regime probabilities (`prob_low`, `prob_med`, `prob_high`) are appended as features to the
main return model (14 features total). Purged CV is re-run and deltas computed.
 
```
accuracy  mean=0.5674  delta vs baseline: +0.0126
auc       mean=0.6011  delta vs baseline: -0.0086
```
 
 
### Per-Regime Models
 
Separate return models are trained per regime subset. At prediction time the active regime
determines which model is queried.
 
| Regime | Bars | Mean Acc | AUC mean | AUC std | Delta vs global |
|--------|------|----------|----------|---------|-----------------|
| low    | 562  | 0.587    | 0.583    | 0.074   | -0.027          |
| medium | 889  | 0.560    | 0.567    | 0.063   | -0.043          |
| high   | 581  | 0.594    | 0.651    | 0.068   | **+0.041**      |
 
Only the high-regime model outperforms the global model. Low and medium regimes lack sufficient signal to justify specialization. This motivates restricting live signals to high-volatility regimes.
 

**Walk-Forward Backtest:**

Full walk-forward simulation with no look-ahead leakage. Parameters:
 
```
min_train     = 600 bars
retrain_every = 50  bars
fee           = 0.05% per side
bar_freq      = 10h
```
 
At each step: HMM is refit on past volatility only, Viterbi decoding is applied only up to
the current bar (not the full dataset), regime classifier and return model are retrained on
past bars. Return model is trained on **high-regime bars only**.
 
### Signal Logic
 
```
in_regime    = prob_high > 0.55
long_entry   = in_regime AND ret_prob > 0.55   (lagged 1 bar)
short_entry  = in_regime AND ret_prob < 0.45   (lagged 1 bar)
sl_stop      = 1.0 * atr_raw / close
tp_stop      = 1.5 * atr_raw / close
```
 
### Output

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total return | 202.63% | -27.71% |
| Sharpe ratio | 2.052 | 0.057 |
| Max drawdown | -27.53% | -64.98% |
| Calmar ratio | 3.516 | — |
| Total trades | 91 | — |
| Win rate | 58.2% | — |
| Avg trade return | 1.353% | — |

### Trade Statistics

| Metric | Value |
|---|---|
| Avg duration (bars) | 4.9 |
| Median duration | 4.0 |
| Return std | 5.08% |
| p25 / p75 return | -3.59% / +5.99% |
| Worst trade | -9.70% |
| Best trade | +10.45% |

![Equity Curve](results/backtest_equity.png)



**Limitations:** 
91 trades over 1.5 years. The benchmark period (Aug 2024 - Mar 2026) 
was a bear market for ETH.    
All hyperparameters (pt_mult, prob_high threshold, retrain_every, max_depth, min_samples_leaf) were selected by iterating on the same period used for backtesting. 
The system as a whole was implicitly optimized on the full data period. 

This backtest is a learning project, not a validated trading system.

---


## Usage
```bash
git clone https://github.com/pynat/causality
pip install -r requirements.txt
jupyter notebook inference_and_causality.ipynb
```