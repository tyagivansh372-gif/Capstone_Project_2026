# TSA Capstone 2026 — Project Skills & Guide

## Project Overview
**Name:** TSA Capstone 2026 — NSE Time Series Forecasting & Portfolio Optimization
**Goal:** Build a production-quality Python project covering stock forecasting, volatility analysis, portfolio optimization, and StockGro virtual trading evaluation.
**Submitted to:** Consulting & Analytics Club, IIT Guwahati
**Trading Platform:** StockGro (virtual ₹10,00,000 portfolio)

## Architecture Map

```
tsa_capstone/
├── state.yaml              ← machine memory; update after every task
├── skills.md               ← project guide; update only on architecture change
├── requirements.txt
├── README.md
├── main.py                 ← orchestration entry point
├── check_env.py            ← import validation script
├── data/
│   ├── raw/                ← yfinance downloads, never modified
│   ├── processed/          ← cleaned, scaled, differenced datasets
│   └── external/           ← any supplementary data
├── notebooks/              ← submission .ipynb files
├── models/                 ← serialized model artifacts (.pkl, .keras, .json)
├── outputs/
│   ├── forecasts/          ← forecast CSVs (backtest + live May 2026)
│   ├── metrics/            ← evaluation CSVs
│   ├── plots/              ← saved figures
│   └── reports/            ← portfolio tables, comparison tables
├── dashboard/              ← Streamlit app entry point
└── src/
    ├── config/             ← config.py: all constants, tickers, date ranges
    ├── data_fetching/
    ├── preprocessing/
    ├── forecasting/
    │   ├── arima.py
    │   ├── sarima.py
    │   ├── holt_winters.py
    │   ├── prophet_model.py
    │   ├── lstm.py
    │   ├── gru.py
    │   ├── garch.py
    │   └── ensemble.py
    ├── volatility/
    ├── portfolio/
    ├── evaluation/
    ├── visualization/
    ├── dashboard/
    └── utils/
```

## Tech Stack

**Core:** Python 3.11+, pandas, numpy, scipy

**Forecasting:**
- statsmodels (ARIMA, SARIMA, Holt-Winters)
- pmdarima (auto_arima for AIC/BIC tuning)
- prophet (Facebook Prophet)
- arch (GARCH)

**ML:** scikit-learn, tensorflow-cpu / keras

**Visualization:** plotly, matplotlib, seaborn

**Dashboard:** streamlit

**Data:** yfinance

## Model List

**Phase 4A — Statistical Models:**
- ARIMA (auto-tuned via pmdarima, residual validation)
- SARIMA (seasonal, m=5 for weekly trading)
- Holt-Winters Exponential Smoothing

**Phase 4B — ML/DL Models:**
- Facebook Prophet (with confidence intervals, Indian holidays)
- LSTM (sequence window 60, 2 layers, dropout 0.2)
- GRU (mirror LSTM architecture)

**Phase 4C — Volatility Model:**
- GARCH(1,1) via `arch` library

**Phase 4D — Ensemble:**
- Weighted average ensemble (ARIMA + Prophet + LSTM)
- Weights from inverse-MAPE on backtest

## Stock Universe

| Stock | Ticker | Sector |
|-------|--------|--------|
| Reliance Industries | RELIANCE.NS | Energy / Conglomerate |
| HDFC Bank | HDFCBANK.NS | Banking |
| Infosys | INFY.NS | Information Technology |
| Sun Pharma | SUNPHARMA.NS | Pharmaceuticals |
| Maruti Suzuki | MARUTI.NS | Automobile |
| ITC | ITC.NS | FMCG |
| Tata Steel | TATASTEEL.NS | Metals |
| Bajaj Finance | BAJFINANCE.NS | NBFC / Financial Services |

## Date Ranges

- **Train:** Jan 1, 2021 → Jun 30, 2025
- **Backtest:** Jul 1, 2025 → Dec 31, 2025
- **Extended:** Jan 1, 2021 → May 10, 2026
- **Live Forecast:** May 11–15, 2026 (5 trading days)

## Coding Standards

- Type hints on all functions
- `logging` (not print) for runtime output
- Exception handling with descriptive messages
- Docstrings (Google style) on classes and public functions
- `config.py` as single source of truth for constants
- PEP8 compliance
- No hardcoded strings outside config

## Current Priorities

**Active Phase:** 1 — Foundation Setup
- Create virtual environment instructions
- Create `requirements.txt` with pinned versions
- Create full folder structure
- Create `config.py` with all constants
- Create `check_env.py` for import validation
- Create boilerplate module stubs
- Create `main.py` orchestration skeleton

## Workflow Rules

1. Read `state.yaml` at every session start
2. Update `state.yaml` after every completed task
3. Keep values short; no prose in YAML
4. Update `skills.md` only when architecture changes
5. No acknowledgment phrases; jump straight to work
