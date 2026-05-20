"""Data preprocessing module.

Handles missing values, stationarity testing, scaling, and train/backtest splits.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

from config.config import STOCKS, DATE_RANGES, PATHS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw stock data for modeling.
    
    Attributes:
        train_data: Dictionary of training DataFrames per stock
        backtest_data: Dictionary of backtest DataFrames per stock
        scalers: Dictionary of fitted MinMaxScalers per stock
        stationarity_results: Dictionary of ADF test results
    """
    
    def __init__(self) -> None:
        """Initialize preprocessor."""
        self.raw_path: Path = Path(PATHS["data_raw"])
        self.processed_path: Path = Path(PATHS["data_processed"])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.backtest_data: Dict[str, pd.DataFrame] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.stationarity_results: Dict[str, Dict] = {}
        
        self.train_cutoff: str = DATE_RANGES["train_end"]
        self.stocks: List[str] = STOCKS
    
    def load_raw_data(self) -> None:
        """Load raw CSV files into memory."""
        for ticker in self.stocks:
            filepath = self.raw_path / f"{ticker.replace('.', '_')}.csv"
            if filepath.exists():
                # First, try to detect if this is a MultiIndex CSV
                with open(filepath, 'r') as f:
                    f.readline()  # Skip first line
                    second_line = f.readline()
                
                # Check if second line contains "Ticker" (MultiIndex format)
                if "Ticker" in second_line:
                    # MultiIndex format - skip first 2 rows, use row 2 as header
                    df = pd.read_csv(filepath, skiprows=[1, 2], index_col=0)
                else:
                    # Standard format
                    df = pd.read_csv(filepath, index_col=0)
                
                # Ensure index is datetime
                df.index = pd.to_datetime(df.index, errors='coerce')
                
                # Clean up column names and ensure numeric
                df.columns = [str(c).strip() for c in df.columns]
                if "Close" in df.columns:
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                
                # Drop any NaN rows
                df = df.dropna()
                
                # Split into train and backtest
                train_mask = df.index <= self.train_cutoff
                self.train_data[ticker] = df[train_mask].copy()
                self.backtest_data[ticker] = df[~train_mask].copy()
                
                logger.info(f"Loaded {ticker}: {len(df[train_mask])} train, {len(df[~train_mask])} backtest")
            else:
                logger.error(f"Raw data not found for {ticker}")
    
    def handle_missing_values(self) -> None:
        """Handle missing values using forward fill, backward fill, then interpolation."""
        for ticker in self.stocks:
            for split_name, data_dict in [
                ("train", self.train_data),
                ("backtest", self.backtest_data)
            ]:
                if ticker in data_dict:
                    df = data_dict[ticker]
                    
                    # Forward fill
                    df = df.ffill()
                    # Backward fill
                    df = df.bfill()
                    # Log interpolation for any remaining
                    for col in df.columns:
                        if df[col].isna().any():
                            df[col] = df[col].interpolate(method="linear")
                    
                    data_dict[ticker] = df
                    
                    nan_remaining = df.isna().sum().sum()
                    if nan_remaining > 0:
                        logger.warning(f"{ticker} {split_name}: {nan_remaining} NaN remaining")
    
    def test_stationarity(self) -> Dict[str, Dict]:
        """Run ADF test for stationarity on closing prices.
        
        Returns:
            Dictionary of ADF test results per stock
        """
        for ticker in self.stocks:
            if ticker not in self.train_data:
                continue
                
            prices = self.train_data[ticker]["Close"].squeeze()
            
            # Run ADF test
            result = adfuller(prices.dropna())
            
            self.stationarity_results[ticker] = {
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < 0.05,
            }
            
            logger.info(
                f"{ticker} ADF p-value: {result[1]:.4f} "
                f"({'stationary' if result[1] < 0.05 else 'non-stationary'})"
            )
        
        return self.stationarity_results
    
    def apply_differencing(self) -> None:
        """Apply first differencing to non-stationary series."""
        for ticker, results in self.stationarity_results.items():
            if not results["is_stationary"]:
                logger.info(f"Applying first differencing to {ticker}")
                
                for data_dict in [self.train_data, self.backtest_data]:
                    if ticker in data_dict:
                        df = data_dict[ticker]
                        df["Close_diff"] = df["Close"].diff()
                        data_dict[ticker] = df
    
    def apply_scaling(self) -> None:
        """Apply MinMaxScaler fitted only on training data (no leakage)."""
        for ticker in self.stocks:
            if ticker not in self.train_data:
                continue
            
            # Fit scaler on training Close prices
            train_prices = self.train_data[ticker]["Close"].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(train_prices)
            
            self.scalers[ticker] = scaler
            
            # Transform both train and backtest
            for data_dict in [self.train_data, self.backtest_data]:
                if ticker in data_dict:
                    df = data_dict[ticker]
                    df["Close_scaled"] = scaler.transform(
                        df["Close"].values.reshape(-1, 1)
                    )
                    data_dict[ticker] = df
    
    def get_scaled_data(
        self, 
        ticker: str, 
        split: str = "train"
    ) -> Optional[np.ndarray]:
        """Get scaled Close prices for a stock.
        
        Args:
            ticker: Stock ticker
            split: 'train' or 'backtest'
            
        Returns:
            Scaled price array
        """
        data_dict = self.train_data if split == "train" else self.backtest_data
        if ticker in data_dict:
            return data_dict[ticker]["Close_scaled"].dropna().values
        return None
    
    def inverse_scale(
        self, 
        ticker: str, 
        scaled_values: np.ndarray
    ) -> np.ndarray:
        """Inverse transform scaled values to original scale."""
        if ticker in self.scalers:
            return self.scalers[ticker].inverse_transform(
                scaled_values.reshape(-1, 1)
            ).flatten()
        return scaled_values
    
    def save_processed_data(self) -> None:
        """Save processed train and backtest splits to disk."""
        for ticker in self.stocks:
            if ticker in self.train_data:
                train_path = self.processed_path / f"{ticker.replace('.', '_')}_train.csv"
                self.train_data[ticker].to_csv(train_path)
            
            if ticker in self.backtest_data:
                backtest_path = self.processed_path / f"{ticker.replace('.', '_')}_backtest.csv"
                self.backtest_data[ticker].to_csv(backtest_path)
        
        # Save stationarity report
        if self.stationarity_results:
            report_df = pd.DataFrame(self.stationarity_results).T
            report_path = self.processed_path / "stationarity_report.csv"
            report_df.to_csv(report_path)
            logger.info(f"Saved stationarity report to {report_path}")
        
        logger.info("Processed data saved successfully")
