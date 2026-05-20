"""Configuration module for TSA Capstone 2026.

Single source of truth for all constants, tickers, date ranges,
and hyperparameter defaults. No hardcoded values elsewhere.
"""

from typing import List, Dict, Any

# =============================================================================
# PROJECT METADATA
# =============================================================================
PROJECT_NAME: str = "TSA Capstone 2026"
PROJECT_VERSION: str = "1.0.0"

# =============================================================================
# STOCK UNIVERSE - NSE Tickers (yfinance format)
# =============================================================================
STOCKS: List[str] = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "SUNPHARMA.NS",
    "MARUTI.NS",
    "ITC.NS",
    "TATASTEEL.NS",
    "BAJFINANCE.NS",
]

# =============================================================================
# COLUMN NAME MAPPINGS
# =============================================================================
COLUMN_NAMES: Dict[str, str] = {
    "ticker": "stock",
}

# =============================================================================
# RISK THRESHOLDS
# =============================================================================
RISK_THRESHOLDS: Dict[str, float] = {
    "high": 0.015,  # > 1.5% volatility
    "medium": 0.010,  # > 1.0% volatility
}

# =============================================================================
# CHART COLOR PALETTE
# =============================================================================
CHART_COLORS: Dict[str, str] = {
    "primary": "#4F8BF9",
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "neutral": "#F39C12",
    "warning": "#F5A623",
    "background": "rgba(0,0,0,0)",
    "grid": "rgba(255,255,255,0.1)",
    "table_header": "#4F8BF9",
    "table_cell_best": "#1A3A2A",
    "table_cell_worst": "#3A1A1A",
    "table_cell_default": "#1A1F2E",
    "live_forecast": "#4F8BF9",
    "backtest": "#F5A623",
    "ci_fill": "rgba(79, 139, 249, 0.15)",
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================
UI_CONFIG: Dict[str, Any] = {
    "portfolio_page_size": 10,
}

# =============================================================================
# HISTORICAL PRICE BOUNDS (INR) — Used for numeric input validation
# Based on approximate 52-week high/low ranges for each NSE stock
# =============================================================================
PRICE_BOUNDS: Dict[str, Dict[str, float]] = {
    "RELIANCE.NS": {"min": 1000.0, "max": 4000.0},
    "HDFCBANK.NS": {"min": 1000.0, "max": 2500.0},
    "INFY.NS": {"min": 1000.0, "max": 2500.0},
    "SUNPHARMA.NS": {"min": 800.0, "max": 2500.0},
    "MARUTI.NS": {"min": 8000.0, "max": 16000.0},
    "ITC.NS": {"min": 200.0, "max": 600.0},
    "TATASTEEL.NS": {"min": 80.0, "max": 250.0},
    "BAJFINANCE.NS": {"min": 4000.0, "max": 12000.0},
}

STOCK_INFO: Dict[str, Dict[str, str]] = {
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy / Conglomerate"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking"},
    "INFY.NS": {"name": "Infosys", "sector": "Information Technology"},
    "SUNPHARMA.NS": {"name": "Sun Pharma", "sector": "Pharmaceuticals"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Automobile"},
    "ITC.NS": {"name": "ITC", "sector": "FMCG"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Metals"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "NBFC / Financial Services"},
}

# =============================================================================
# DATE RANGES (YYYY-MM-DD format)
# =============================================================================
DATE_RANGES: Dict[str, str] = {
    "train_start": "2021-01-01",
    "train_end": "2025-06-30",
    "backtest_start": "2025-07-01",
    "backtest_end": "2025-12-31",
    "extended_train_end": "2026-05-10",
    "live_forecast_start": "2026-05-11",
    "live_forecast_end": "2026-05-15",
}

# =============================================================================
# DIRECTORIES (relative to project root)
# =============================================================================
PATHS: Dict[str, str] = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "data_external": "data/external",
    "models": "models",
    "outputs_forecasts": "outputs/forecasts",
    "outputs_metrics": "outputs/metrics",
    "outputs_plots": "outputs/plots",
    "outputs_reports": "outputs/reports",
    "notebooks": "notebooks",
    "dashboard": "dashboard",
}

# =============================================================================
# FORECASTING HYPERPARAMETERS
# =============================================================================
ARIMA_PARAMS: Dict[str, Any] = {
    "start_p": 1,
    "start_q": 1,
    "max_p": 5,
    "max_q": 5,
    "min_p": 1,
    "min_q": 1,
    "max_d": 2,
    "d": None,
    "seasonal": False,
    "information_criterion": "aic",
    "error_action": "ignore",
    "suppress_warnings": True,
    "stepwise": True,
}

SARIMA_PARAMS: Dict[str, Any] = {
    "seasonal": True,
    "m": 5,  # Weekly seasonality for trading days
    "stepwise": True,
    "suppress_warnings": True,
    "start_p": 1,
    "start_q": 1,
    "max_p": 3,
    "min_p": 1,
    "max_q": 3,
    "min_q": 1,
    "max_d": 2,
    "max_P": 2,
    "max_D": 1,
    "max_Q": 2,
}

HOLT_WINTERS_PARAMS: Dict[str, Any] = {
    "trend": "add",
    "damped_trend": False,
    "seasonal": None,
    "initialization_method": "estimated",
}

PROPHET_PARAMS: Dict[str, Any] = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.95,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
}

# LSTM/GRU shared parameters
DEEP_LEARNING_PARAMS: Dict[str, Any] = {
    "sequence_length": 60,
    "lstm_units": [64, 32],
    "gru_units": [64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 15,
    "validation_split": 0.1,
}

GARCH_PARAMS: Dict[str, Any] = {
    "p": 1,
    "q": 1,
    "vol": "GARCH",
    "dist": "normal",
}

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================
PORTFOLIO_CONFIG: Dict[str, Any] = {
    "total_capital": 1000000.0,  # INR 10,00,000
    "max_position_pct": 0.25,  # Max 25% per stock
    "min_position_pct": 0.05,  # Min 5% per stock
    "volatility_lookback": 90,
    "correlation_lookback": 90,
    "momentum_lookback": 126,  # 6 months
}

# =============================================================================
# EVALUATION METRICS
# =============================================================================
METRICS: List[str] = [
    "rmse",
    "mae",
    "mape",
    "directional_accuracy",
]

# =============================================================================
# LOGGING CONFIG
# =============================================================================
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# =============================================================================
# RANDOM SEED (Reproducibility)
# =============================================================================
RANDOM_SEED: int = 42
