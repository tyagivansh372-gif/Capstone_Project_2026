"""Data loading utilities for Streamlit dashboard.

All data loading functions use @st.cache_data decorator to prevent
re-reading files on every interaction.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Bootstrap: add src/ to sys.path so config and src modules are importable
sys.path.insert(0, "src")

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def load_portfolio_allocation() -> pd.DataFrame:
    """Load portfolio allocation from CSV.
    
    Returns:
        pd.DataFrame: DataFrame with portfolio allocation data containing
            ticker, weight_pct, allocated_INR, shares_to_buy, and last_price columns.
            Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("outputs/reports/portfolio_allocation.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = ["ticker", "weight_pct", "allocated_INR", "shares_to_buy", "last_price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in portfolio_allocation.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_model_comparison() -> pd.DataFrame:
    """Load model comparison metrics from CSV.
    
    Returns:
        pd.DataFrame: DataFrame with model comparison data containing
            stock, model, mape, rmse, mae, and directional_accuracy columns.
            Column names are normalized to lowercase and 'ticker' is renamed to 'stock'.
            Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    from config.config import COLUMN_NAMES
    
    path = Path("outputs/metrics/model_comparison.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [col.lower() for col in df.columns]
    # Apply column name mappings from config
    for old_name, new_name in COLUMN_NAMES.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    required_cols = ["stock", "model", "mape", "rmse", "mae", "directional_accuracy"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in model_comparison.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_live_forecasts() -> pd.DataFrame:
    """Load live forecasts for May 2026.
    
    Returns:
        pd.DataFrame: DataFrame with live forecast data containing
            stock, date, predicted_price, and model columns. Date column
            is parsed as datetime. Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("outputs/forecasts/live_forecasts_may2026.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = ["stock", "date", "predicted_price", "model"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in live_forecasts_may2026.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_garch_volatility() -> pd.DataFrame:
    """Load GARCH volatility forecasts.
    
    Returns:
        pd.DataFrame: DataFrame with GARCH volatility data containing
            stock, date, and forecasted_volatility columns. Date column
            is parsed as datetime. Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("outputs/forecasts/garch_volatility_may2026.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = ["stock", "date", "forecasted_volatility"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in garch_volatility_may2026.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_live_actuals() -> pd.DataFrame:
    """Load live actuals (may contain nulls).
    
    Returns:
        pd.DataFrame: DataFrame with live actuals data containing
            stock, date, actual_close_price, and actual_return_pct columns.
            Date column is parsed as datetime. May contain null values.
            Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("data/external/live_actuals_may2026.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = ["stock", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in live_actuals_may2026.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_live_vs_predicted() -> pd.DataFrame:
    """Load live vs predicted comparison.
    
    Returns:
        pd.DataFrame: DataFrame with live vs predicted data containing
            stock, date, predicted_price, and status columns. Date column
            is parsed as datetime. Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("outputs/reports/live_vs_predicted.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = ["stock", "date", "predicted_price", "status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in live_vs_predicted.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def load_backtest_forecasts(stock: str) -> pd.DataFrame:
    """Load backtest forecasts for a specific stock.
    
    Args:
        stock: Stock ticker symbol
        
    Returns:
        DataFrame with backtest forecasts (empty if file missing)
    """
    safe = stock.replace(".", "_")
    # Try multiple naming patterns
    paths = [
        Path(f"outputs/forecasts/backtest_forecasts_{safe}.csv"),
        Path("outputs/forecasts/backtest_forecasts_jul_dec2025.csv"),
    ]

    for path in paths:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            # Filter by stock if multi-stock file
            if "stock" in df.columns:
                df = df[df["stock"] == stock]
            return df

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_raw_prices(stock: str) -> Optional[pd.DataFrame]:
    """Load raw processed price data for a stock.
    
    Args:
        stock: Stock ticker symbol
        
    Returns:
        DataFrame with Close column and DatetimeIndex, or None if not found
    """
    safe = stock.replace(".", "_")
    path = Path(f"data/processed/{safe}_train.csv")

    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")

    # Ensure Close column exists
    if "Close" not in df.columns:
        return None

    return df[["Close"]]


@st.cache_data(ttl=300)
def load_portfolio_performance() -> pd.DataFrame:
    """Load portfolio performance data.
    
    Returns:
        pd.DataFrame: DataFrame with portfolio performance data containing
            stock, weight_pct, status, and actual_return_pct columns.
            Returns empty DataFrame if file not found.
            
    Raises:
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path("outputs/reports/portfolio_performance.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = ["stock", "weight_pct", "status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in portfolio_performance.csv: {missing}")
    return df


@st.cache_data(ttl=300)
def get_stock_list() -> list:
    """Get list of available stocks from config.
    
    Returns:
        list: List of stock ticker strings (e.g., ['RELIANCE.NS', 'HDFCBANK.NS', ...]).
    """
    from config.config import STOCKS
    return STOCKS
