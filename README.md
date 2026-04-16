# Gamma-Scalping-Algorithmic-Trading-Engine

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBM%20Classifier-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-HPT%20Tuning-6C63FF?style=flat)
![Data](https://img.shields.io/badge/Data-2M%2B%20rows-0D6E6E?style=flat)

> **An end-to-end algorithmic options trading engine for Nifty50 weekly expiry,
> combining gamma scalping via ATM straddles with an ML signal filter trained
> using a strict 12-3-3 chronological walk-forward split.**

---

## Table of Contents

- [Overview](#overview)
- [Strategy](#strategy)
- [Architecture](#architecture)
- [Engine Versions](#engine-versions)
- [ML Pipeline](#ml-pipeline)
- [Features](#features)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This project implements a systematic options strategy on **Nifty50 weekly
expiry contracts** using 18 months of 1-minute tick data (2024-08-26 →
2026-02-16, 2,013,680 rows). The core idea: buy an ATM straddle when implied
volatility is cheap relative to realised volatility, capture directional moves
through delta-hedging, and filter entry signals with a Gradient Boosting
Machine (GBM) classifier.

The engine went through **5 major versions**  each iteration uncovering and
fixing a deeper category of bug before arriving at the final ML-enhanced
pipeline.

---

## Strategy

### Core Mechanism — Gamma Scalping

```
Gamma P&L = 0.5 × Γ × (σ_realised² − σ_implied²) × S² × dt
```

1. **Buy ATM straddle** (CE + PE at the nearest 50-pt strike to spot)
   when IV Rank < 50 and IV/RV < 1.037
2. **Delta-hedge** every time spot moves ≥ 51 points from the last hedge
   level buy PUT on up-moves, buy CALL on down-moves
3. **Close hedge** when spot reverts within 41 points (captures the
   gamma round-trip)
4. **Exit straddle** on one of five triggers:

| Exit | Trigger | Direction |
|------|---------|-----------|
| `gamma_win` | Straddle +15.5% | Take-profit |
| `vega_win` | IV expanded +10% | Take-profit |
| `max_drawdown` | Straddle −13.8% | Stop-loss |
| `theta_bleed` | 88 min elapsed + below entry | Time-stop |
| `eod` | 15:10 IST | Force-close |

### Entry Filters (Optuna-tuned)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `IVR_MAX` | 49.79 | Enter only when IV is in lower 50% of its 23-day range |
| `IV_RV_MAX` | 1.037 | Realised vol must exceed implied vol (edge condition) |
| `RV_WINDOW` | 59 bars | ~1 hour of intraday vol (Optuna: 46% importance) |
| `IVR_DAYS` | 23 days | Rolling IVR lookback ≈ 1 month |
| `DTE_MIN/MAX` | 2–4 days | Balance gamma vs theta; avoid expiry-day chaos |
| `ENTRY_START` | 09:30 IST | — |
| `ENTRY_END` | 12:00 IST | Afternoon IV events rare; theta drag accumulates |

---

## Architecture

```
1-MIN option data (2M+ rows)
        │
        ▼
┌─────────────────────┐
│  ATM Straddle Build │  ← merge CE + PE at offset=0
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Indicators         │  IV, RV (59-bar), IVR, IV/RV, ER, DTE, ATR
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Signal Filter      │  IVR < 49.79  &  IV/RV < 1.037  &  DTE 2-4
└────────┬────────────┘
         │ signal bars
         ▼
┌─────────────────────┐
│  Feature Engineering│  25 features — zero lookahead
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 LABEL    12-3-3 SPLIT
 ENGINE   (chronological)
    │         │
    │    TRAIN│VALID│TEST
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────────┐
│  GBM Classifier     │  200-iter RandomizedSearch  ×  StratifiedKFold-5
│  (≡ XGBoost)        │  Threshold tuned on VALID only
└────────┬────────────┘
         │ P(win) ≥ threshold
         ▼
┌─────────────────────┐
│  ML-Gated Backtest  │  v4 engine unchanged — ML gate at entry only
└─────────────────────┘
```

---

## Engine Versions

| Version | Trades | Total P&L | Sharpe | Key Flaw Fixed |
|---------|--------|-----------|--------|----------------|
| v1 | 224 | −₹787,883 | -6.44 | Concept only — no implementation |
| v2 | 135 | −₹21,341 | −2.67 | First engine; hedge silent failure; iv_crush bug |
| v3 | 18 | −₹10,301 | −7.29 | Datetime mismatch; over-filtered (18 trades) |
| v4 | 46 | −₹7,343 | −0.93 | Inverted hedge direction; 22-min RV vs daily IV |
| v5-ML | 239 | Pipeline complete | — | BS gamma hedge; daily RV; ML gate |

### Bug History

| # | Version | Bug | Impact |
|---|---------|-----|--------|
| 1 | v2 | Option lookup silent failure → hedge_pnl = 0 | Zero gamma scalping executed |
| 2 | v2 | `iv_crush` exit at 4.3% IV drop | −₹72,754 (59 premature exits) |
| 3 | v2 | 09:30 entries hit IV compression | −₹12,806 (67 trades) |
| 4 | v2 | No weekday filter | −₹8,550 (Saturday trades; NSE closed) |
| 5 | v3 | 1-min dt never in 3-min set → hedge gate blocks all | hedge_pnl = 0 again |
| 6 | v3 | IV/RV < 0.606 too tight | Only 18 trades in 18 months |
| 7 | v3 | RV_WINDOW = 22 bars = 22 min (microstructure) | RV spikes to 183%; false edge signal |
| 8 | v4 | Hedge buys ITM option after spot move (inverted) | −₹70,749 hedge losses |
| 9 | v4 | THETA_MAX_MIN = 180 min | −₹134,188 theta_bleed losses (76% of total) |
| 10 | v4 | No ML filter | Low- and high-quality signals treated equally |

---

## ML Pipeline

### Feature Engineering (25 features, zero lookahead)

| Group | Features |
|-------|----------|
| Volatility (5) | `iv_pct`, `rv_pct`, `iv_rv_ratio`, `ivr`, `iv_minus_rv` |
| Structure (4) | `er`, `dte`, `straddle_pct_spot`, `straddle_z_20d` |
| Time (5) | `hour`, `minute`, `dow`, `month`, `week_of_month` |
| Momentum (6) | `spot_ret_30m`, `spot_ret_60m`, `iv_chg_15m`, `iv_chg_30m`, `straddle_chg_30m`, `spot_atr_ratio` |
| Interactions (5) | `ivr_sq`, `dte_ivr`, `er_ivr`, `iv_rv_ivr`, `straddle_z_ivr` |

### Chronological 12-3-3 Split

```
Aug 2024 ──────────────── Aug 2025 ─── Nov 2025 ─── Feb 2026
│◄──────── TRAIN (12 mo) ────────►│◄── VALID ──►│◄── TEST ──►│
                                       (3 mo)       (3 mo)
```

- **Train**: model learns feature → outcome mapping
- **Valid**: threshold tuned (precision ≥ 55%, recall ≥ 18%)
- **Test**: fully unseen; only number cited in results

### Hyperparameter Tuning

```python
param_space = {
    'n_estimators'         : randint(150, 900),
    'max_depth'            : randint(2, 7),
    'learning_rate'        : loguniform(0.003, 0.25),
    'subsample'            : uniform(0.5, 0.5),
    'max_features'         : uniform(0.4, 0.6),
    'min_samples_leaf'     : randint(5, 80),
    'min_impurity_decrease': loguniform(1e-6, 5e-3),
    'ccp_alpha'            : loguniform(1e-6, 5e-3),
}

```

---

## Features

- **Full 1-min backtesting engine** — tick-level entry/exit, hedge tracking,
  cost deduction, daily trade limits, cooldown timers
- **Black-Scholes delta hedge** — no option price lookup; gamma P&L from
  `0.5 × Γ × Δspot² × lots × lot_size`
- **Daily realised vol** — close-to-close 5-day RV annualised;
  correctly comparable to option-implied annual IV
- **25-feature ML layer** — GBM classifier, zero lookahead, class-weight
  balanced, threshold tuned on hold-out validation
- **Strict walk-forward split** — chronological 12-3-3; no shuffling;
  test set never touched until final evaluation
- **Full Optuna-equivalent HPT** — 200 randomised trials, log-uniform
  sampling for learning rate and regularisation params
- **Comprehensive reporting** — equity curves, drawdown, exit-reason heatmap,
  PR-curve, feature importance, split-by-split metrics table

---

## Dataset

| Property | Value |
|----------|-------|
| Underlying | Nifty50 Index Options (NSE) |
| Frequency | 1-minute bars |
| Date range | 2024-08-26 → 2026-02-16 |
| Total rows | 2,013,680 (7 CSV parts) |
| Strikes | ATM ± 2 (strike_offset filter) |
| Option types | CALL, PUT |
| Columns | datetime, open, high, low, close, iv, spot, volume, oi, option_type, strike_offset |

---

## Results

### v4 Baseline (no ML)

| Metric | Value |
|--------|-------|
| Trades | 46 |
| Win Rate | 34.8% |
| Total P&L | −₹7,343 |
| Profit Factor | 0.85 |
| Sharpe | −0.93 |
| Max Drawdown | −₹12,378 |
| Avg Hedges/Trade | 1.9 |

### v5-ML (with GBM gate)

ML pipeline operational. Sample size constraint (47 labels with DTE 3-4 filter)
prevents full walk-forward reporting. Expanding DTE filter to 2-6 generates
~135 labelled trades — sufficient for the 12-3-3 split to produce statistically
meaningful VALID and TEST results.

---

## Installation

```bash
git clone https://github.com/your-username/Gamma-Scalping-Algorithmic-Trading-Engine.git
cd Gamma-Scalping-Algorithmic-Trading-Engine

pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.26
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.8
```

---

## Usage

### Run v4 baseline engine

```python
from gamma_engine_v4 import run

trades = run(
    data_1min = 'data/1MIN',
    out_dir   = 'output/v4',
)
```

### Run ML pipeline (v5)

```python
from gamma_ml_v5 import run_ml_pipeline

results = run_ml_pipeline(
    data_1min        = 'data/1MIN',
    out_dir          = 'output/ml',
    n_hpt_iter       = 200,
    target_precision = 0.55,
    min_recall       = 0.18,
    seed             = 42,
)

model     = results['model']
threshold = results['threshold']
test_pnl  = results['test_trades']['realized_pnl'].sum()
```

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

from gamma_ml_v5 import run_ml_pipeline
results = run_ml_pipeline(
    data_1min = '/content/drive/MyDrive/1MIN',
    out_dir   = '/content/drive/MyDrive/gamma_ml_output',
)
```

---

## Project Structure

```
GAMMA-SCALPING-ALGORITHMIC-TRADING-ENGINE/
├── BACKTEST RESULTS/
│   ├── backtest_v1.png
│   ├── backtest_v2.png
│   ├── backtest_v3.png
│   ├── backtest_v4.png
│   ├── backtest_v5.png
│   └── ml_backtest_v5.png
├── BEST PARAMETERS/
│   ├── finance_best_params.json
│   └── ml_best_params.json
├── DATA/
│   ├── 1MIN/
│   ├── 3MIN/
│   └── 5MIN/
├── EDA/
│   ├── eda_report_1.png
│   ├── eda_report_2.png
│   └── eda_report.png
├── OPTUNA RESULTS/
├── TRADE LOGS/
│   ├── trade_logs_v4.csv
│   └── trade_logs_v5_ml.csv
├── .gitattributes
├── README.md
├── report.html
├── Round_2_Final.ipynb
└── strat.txt
```

---
