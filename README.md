# UNDER CONSTRUCTION

# De Prado ML Framework with Causal Discovery: Regime-Model on ETH

## Overview

Marcos López de Prado argues that most machine learning models in finance fail not because of model complexity, but due to **spurious features**, statistical relationships that do not reflect the true data-generating process.

This project implements de Prado's ML framework on ETH/USDT, extended with causal discovery as a methodological guardrail. The system predicts **volatility regimes** for Volatility targeting, Strategy switching, Risk overlays.

To address limitations of time-based sampling, the analysis is conducted on **dollar bars** ($500M threshold), which sample observations based on traded value. This reduces heteroskedasticity, improves serial independence of returns, and aligns data with actual market activity.

---

### Research Question

*DDoes causal feature selection improve the predictability of volatility regimes in ETH dollar bars under purged cross-validation?*

### Methodology

- **Dollar Bar Construction** $500M threshold, 2431 bars over 3 years, validated via Durbin-Watson (1.95) and Ljung-Box (p > 0.4)
- **Feature Engineering** volatility, volume, drawdown, technical and tail-risk features, all ADF-tested for stationarity
- **Correlation Analysis** multicollinearity hygiene via clustering (threshold 0.85), not feature selection
- **Causal Discovery** PCMCI map feature dependency structure and identify true drivers vs downstream nodes
- **Triple Barrier Labeling** PT = 1.5x ATR, SL = 1.0x ATR, max hold = 20 bars
- **Random Forest with Purged K-Fold CV** n=5 folds, embargo=1%, prevents temporal leakage
- **MDI/MDA Feature Importance** cross-referenced with causal graph to validate signal vs noise


### Key Finding

Causal discovery and MDI/MDA produced **mutually explanatory results**. RSI ranked 3rd by MDI (in-sample importance) but last by MDA (out-of-sample). PCMCI explained the mechanism: RSI is a pure downstream node driven by `vwap_distance`, `bb_width`, and `drawdown`, it carries no independent predictive signal. MDI was deceived by its correlation with genuine drivers. This cross-validation is the primary methodological contribution of the hybrid framework.

### Results, annualized, out-of-sample, no transaction costs

## Model Performance

### Regime Distribution
- low:   27.9%
- med:   28.7%
- high:  43.4%

### Baseline vs Model
- Baseline Accuracy (majority class): 43.41%
- Model Accuracy (purged CV):         64.15%
- Lift over baseline:                 +20.73%

### Cross-Validation Performance (Purged CV, 5 folds)
- Accuracy: 0.6415 ± 0.024

**AUC (one-vs-rest):**
- low:  0.8651 ± 0.0250
- med:  0.7308 ± 0.0223
- high: 0.9114 ± 0.0147

### Regime Persistence
Transition matrix (P(next | current)):

|       | →low | →med | →high |
|------|------|------|-------|
| low  | 0.717| 0.260| 0.023 |
| med  | 0.173| 0.670| 0.157 |
| high | 0.012| 0.247| 0.741 |

Average persistence: 70.9%

### Feature Importance (mean across folds)
- market_stress:      0.3860
- volatility_7b:      0.2688
- bb_position:        0.1118
- vol_persistence:    0.1041

---

The model shows strong out-of-sample performance, outperforming the baseline and capturing persistent volatility regimes in ETH dollar bars. Predictive power is highest in low- and high-volatility states.


---

## ETH Distribution Characteristics

- **Neutral with negligible positive bias mean return** (0.03%) with median (0.06%) indicates weak bullish drift in the sample period
- **Moderate volatility** (2.22% per bar), more stable than time bars due to event-based sampling
- **Near-symmetric distribution** (skewness -0.03) indicates no strong directional asymmetry
- **Low excess kurtosis** (1.34) suggests extreme events typical for crypto returns

![Return Distribution](results/return_distribution_dollar_bars.png)

### Tail Risk Profile

- **VaR 95%: -3.68%** moderate downside risk per dollar-activity event
- **VaR 99%: -5.45%** rare but meaningful tail losses under high-activity regimes

### Serial Independence Validation

Dollar bars are theoretically more i.i.d. than time bars. This was validated empirically:

- **Durbin-Watson: 1.96** near-perfect, no lag-1 autocorrelation
- **Ljung-Box lag 10: p = 0.41** no autocorrelation across first 10 lags
- **Ljung-Box lag 20: p = 0.70** holds across 20 lags

This confirms the core motivation for dollar bars: returns are statistically independent, satisfying the assumption most ML models require.

![Stationarity](results/autocorrelation_dollar_bars.png)

---

## Feature Engineering

Features are grouped by domain and mapped to distributional properties of the return series:

| Domain | Features | Rationale |
|---|---|---|
| Volatility | `volatility_7b`, `vol_momentum`, `vol_expansion` | Activity-conditioned regime signals |
| Risk | `drawdown`, `deep_drawdown` | Tail risk per unit of traded value |
| Extremes | `extreme_down`, `extreme_streak` | Non-linear shock detection |
| Technical | `bb_width`, `atr_normalized`, `vwap_distance` | Latent market state proxies |
| Volume | `volume_change`, `volume_zscore` | Information flow and activity intensity |

All 30 features passed ADF stationarity testing. Binary features were validated for class balance, 4 features with under 3% positive class were dropped as uninformative.

---

## Correlation Analysis

![Correlation Features](results/correlation.png)

Correlation analysis is used to identify redundant feature groups and control multicollinearity. It serves as a structural filter, not a measure of predictive importance. Feature ranking and final selection are performed post-training via MDI/MDA.

One cluster was identified at a threshold of 0.85: `vwap_distance`, `rsi`, `bb_position` (max correlation 0.93). All three features encode normalized price location relative to a reference (VWAP, momentum oscillator, volatility bands), resulting in high redundancy.


---
## Causal Discovery (PCMCI)

Causal discovery PCMCI (Tigramite, ParCorr, α=0.05, lags 1–5) was used to map directional dependencies between features, identifying true drivers vs. downstream nodes before model training and preventing spurious correlations and look-ahead bias by construction.

**Purpose:** Cross-validate MDI/MDA importance scores with structural causal evidence. Features that rank high in MDI but are causally downstream are flagged as noise candidates.


Key findings:
- `vwap_distance`, `rsi`, and `bb_position` are the strongest causal drivers 
  of `drawdown` (lag=1, strengths: 0.94, 0.90, 0.82)
- `atr_normalized` drives both `drawdown` (0.67) and `extreme_streak` (0.52) 
  at lag=1, volatility expansion predicts extreme bar sequences
- `bb_width` drives volatility, momentum, drawdown, and vwap_distance 
  across multiple lags, most connected upstream node
- 293 significant causal links identified; no contradictions with temporal ordering
- Results used as structural prior for MDI/MDA feature importance interpretation



![PCMCI Algorithm With Strongest Edges](results/pcmci_dag.png)


---

## Triple Barrier Labeling & Feature Importance

**Triple Barrier Labeling** (de Prado) replaces naive return-direction labels with structurally sound targets:

- **Upper barrier:** Profit target at 1.5x ATR
- **Lower barrier:** Stop-loss at 1.0x ATR
- **Vertical barrier:** Maximum hold of 20 bars

Label distribution: 52.5% stops hit, 47.2% profit targets hit, 0.3% timeouts, confirming ETH is volatile enough to always resolve within 20 bars. The near-balanced distribution eliminates the need for resampling.

**MDI vs MDA: The Core Diagnostic**

![Feature Importance](results/random_forest.png)

A Random Forest was trained on all stationary features with triple barrier labels as target. Two importance measures were computed and cross-referenced with causal discovery results:

- **MDI (Mean Decrease Impurity):** In-sample, computed from tree structure. Fast but biased toward high-cardinality and correlated features.
- **MDA (Mean Decrease Accuracy):** Out-of-sample permutation importance on held-out data. Slower but honest.

The gap between MDI and MDA revealed the key finding of this project:

| Feature | MDI Rank | MDA Rank | Verdict |
|---|---|---|---|
| `rsi` | 3 | 25 (last) | noise, causally downstream |
| `volume` | 1 | 24 | noise, no causal confirmation |
| `atr_normalized` | 2 | 1 | genuine signal |
| `bb_width` | 5 | 2 | genuine signal |

9 features with negative MDA were dropped. These features hurt out-of-sample performance despite appearing important in-sample.

**Final Feature Set (MDI/MDA validated, causally confirmed):**

| Feature | MDA Rank | Causal Status |
|---|---|---|
| `atr_normalized` | 1 | driver of vol_regime, volume_zscore |
| `bb_width` | 2 | largest causal hub |
| `volume_change` | 3 | driver of vol_regime, volatility_7b |
| `drawdown` | 4 | driver of rsi, volatility_7b |
| `volatility_7b` | 6 | causal mediator |
| `vol_regime` | 7 | confirmed driver|

**Purged K-Fold Cross Validation**

Standard k-fold leaks future information at fold boundaries due to rolling feature windows. Purged CV removes training samples whose window overlaps the test period, plus an embargo buffer of 1% of bars after each test fold.

| Fold | Accuracy | AUC |
|---|---|---|
| 1 | 0.544 | 0.543 |
| 2 | 0.533 | 0.536 |
| 3 | 0.536 | 0.524 |
| 4 | 0.525 | 0.529 |
| 5 | 0.525 | 0.534 |
| **mean** | **0.533** | **0.533** |

AUC = 0.533 on triple barrier labels, weak but present edge. This is the honest baseline before regime conditioning. The regime model achieves substantially higher AUC (0.84–0.91) because volatility persistence is a structurally easier prediction problem than price direction.



---

## Usage
```bash
git clone https://github.com/pynat/causality
pip install -r requirements.txt
jupyter notebook inference_and_causality.ipynb
```