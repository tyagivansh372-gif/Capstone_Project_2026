# TSA Capstone 2026 — NSE Time Series Forecasting & Portfolio Optimization

**Submitted to:** Consulting & Analytics Club, IIT Guwahati  
**Trading Platform:** StockGro (virtual ₹10,00,000 portfolio)  
**Live Trading Window:** May 11-15, 2026

---

## Project Overview

This capstone project implements a production-quality time series forecasting pipeline for 8 NSE stocks, covering:

- **Data Pipeline:** Historical data (2021-2025) + extended data (through May 10, 2026)
- **7 Forecasting Models:** ARIMA, SARIMA, Holt-Winters, Prophet, LSTM, GRU, GARCH
- **Ensemble Methods:** Weighted average ensemble (ARIMA + Prophet + LSTM)
- **Portfolio Optimization:** 4-strategy allocation framework
- **Interactive Dashboard:** Streamlit app for visualization and live actuals entry

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Environment

```bash
python check_env.py
```

---

## Project Structure

```
tsa_capstone/
├── state.yaml              # Machine-readable project state
├── skills.md               # Project guide and architecture
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── main.py                 # Orchestration entry point
├── check_env.py            # Environment validation
│
├── data/
│   ├── raw/                # yfinance downloads
│   ├── processed/          # Cleaned, scaled datasets
│   └── external/           # Live actuals (manual entry)
│
├── notebooks/              # Submission notebook
│
├── models/                 # Serialized models (.pkl, .keras)
│
├── outputs/
│   ├── forecasts/          # All forecasts (CSV)
│   ├── metrics/            # Evaluation metrics (CSV)
│   ├── plots/              # Visualization images (PNG)
│   └── reports/            # Summary tables (CSV, MD)
│
├── dashboard/              # Streamlit app
│   └── app.py
├── .streamlit/
│   └── config.toml         # Dark theme configuration
│
└── src/
    ├── config/
    │   └── config.py       # All constants and parameters
    ├── data_fetching/
    │   └── fetcher.py
    ├── preprocessing/
    │   └── preprocessor.py
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
    │   └── analyzer.py
    ├── portfolio/
    │   └── optimizer.py
    ├── evaluation/
    │   └── evaluator.py
    ├── visualization/
    │   └── plots.py
    ├── dashboard/
    └── utils/
        ├── reporter.py
        └── notebook_compiler.py
```

---

## Running the Pipeline

### Run Complete Pipeline
```bash
python main.py
```

### Run Individual Phases
```bash
python main.py --phase 2    # Data fetching
python main.py --phase 3    # Preprocessing
python main.py --phase 4a   # Statistical models
python main.py --phase 4b   # ML/DL models
python main.py --phase 4c   # GARCH volatility
python main.py --phase 4d   # Ensemble + live forecasts
python main.py --phase 5    # Volatility analysis
python main.py --phase 6    # Portfolio optimization
python main.py --phase 7    # Evaluation
```

---

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard features:
- Stock selector with forecast visualization
- Model comparison tables (sortable)
- Portfolio allocation charts
- Correlation heatmap
- Volatility plots
- **Live actuals entry form** for StockGro data
- Downloadable reports (CSV)

---

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

---

## Key Outputs

### Forecasts
- `outputs/forecasts/live_forecasts_may2026.csv` — 5-day price forecasts
- `outputs/forecasts/garch_volatility_may2026.csv` — Conditional volatility

### Metrics
- `outputs/metrics/model_comparison.csv` — All models vs all metrics
- `outputs/metrics/ensemble_weights.csv` — Inverse-MAPE weights

### Reports
- `outputs/reports/portfolio_allocation.csv` — Final ₹10L allocation
- `outputs/reports/portfolio_performance.csv` — Backtest results
- `outputs/reports/live_vs_predicted.csv` — Post-trading comparison (manual entry)
- `outputs/reports/final_summary.md` — Complete project summary

### Notebook
- `notebooks/capstone_submission.ipynb` — Auto-generated submission

---

## Date Ranges

| Period | Dates |
|--------|-------|
| Train | Jan 1, 2021 → Jun 30, 2025 |
| Backtest | Jul 1, 2025 → Dec 31, 2025 |
| Extended | Jan 1, 2021 → May 10, 2026 |
| Live Forecast | May 11-15, 2026 |

---

## Model Configuration

### ARIMA
- Auto-tuned via `pmdarima`
- Max p=5, d=2, q=5
- AIC-based selection
- Residual Ljung-Box test

### SARIMA
- Weekly seasonality (m=5)
- Same tuning as ARIMA

### Prophet
- Yearly + weekly seasonality
- Indian market holidays
- 95% confidence intervals

### LSTM/GRU
- Sequence length: 60 days
- 2 layers (64 → 32 units)
- Dropout: 0.2
- Early stopping: 15 epochs patience

### GARCH(1,1)
- Normal distribution
- Log returns input
- 5-day volatility forecast

### Ensemble
- ARIMA + Prophet + LSTM
- Weights: inverse-MAPE from backtest
- Normalized to sum = 1

---

## Evaluation Metrics

All models evaluated on:
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **Directional Accuracy** — % correct direction predictions

---

## Portfolio Strategies

| Strategy | Description | Weight |
|----------|-------------|--------|
| A | Forecast-guided (rank by predicted return) | 60% |
| B | Volatility-aware (inverse-volatility) | 40% |
| C | Correlation-based (diversification penalty) | Validation |
| D | Sector momentum rotation | Validation |

---

## Post-Trading Evaluation

After trading on StockGro (May 11-15, 2026):

1. Open the dashboard: `streamlit run dashboard/app.py`
2. Navigate to "Live Actuals Entry"
3. Enter closing prices for each stock and date
4. Click "Save Actuals"
5. View prediction accuracy (MAPE, directional accuracy)
6. Check `outputs/reports/live_vs_predicted.csv` for results

---

## Tech Stack

- **Python 3.11+**
- **Core:** pandas, numpy, scipy
- **Forecasting:** statsmodels, pmdarima, prophet, arch
- **ML/DL:** scikit-learn, tensorflow-cpu
- **Data:** yfinance
- **Viz:** plotly, matplotlib, seaborn
- **Dashboard:** streamlit
- **Notebook:** nbformat

---

## License

Academic project — Consulting & Analytics Club, IIT Guwahati
