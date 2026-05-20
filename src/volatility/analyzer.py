"""Volatility and trend analysis module.

Computes rolling volatility, STL decomposition, and trend classification.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from config.config import STOCKS, PATHS
from preprocessing.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """Analyzes volatility and trends for stock universe.
    
    Attributes:
        volatility_data: Dictionary of rolling volatility series
        stl_results: Dictionary of STL decomposition results
        trend_classifications: Dictionary of trend labels
    """
    
    def __init__(self) -> None:
        """Initialize analyzer."""
        self.volatility_data: Dict[str, Dict[str, pd.Series]] = {}
        self.stl_results: Dict[str, object] = {}
        self.trend_classifications: Dict[str, str] = {}
        
        self.reports_path: Path = Path(PATHS["outputs_reports"])
        self.reports_path.mkdir(parents=True, exist_ok=True)
    
    def compute_rolling_volatility(self, windows: List[int] = [30, 90]) -> None:
        """Compute rolling standard deviation (volatility) for all stocks.
        
        Args:
            windows: List of lookback windows in days
        """
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker not in preprocessor.train_data:
                continue
            
            prices = preprocessor.train_data[ticker]["Close"].squeeze()
            returns = prices.pct_change().dropna()
            
            self.volatility_data[ticker] = {}
            
            for window in windows:
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
                self.volatility_data[ticker][f"vol_{window}d"] = rolling_vol
    
    def stl_decomposition(self, period: int = 5) -> None:
        """Perform STL decomposition for all stocks.
        
        Args:
            period: Seasonal period (default 5 for weekly)
        """
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker not in preprocessor.train_data:
                continue
            
            prices = preprocessor.train_data[ticker]["Close"].squeeze()
            
            try:
                stl = STL(prices, period=period, robust=True)
                result = stl.fit()
                
                self.stl_results[ticker] = {
                    "trend": result.trend,
                    "seasonal": result.seasonal,
                    "resid": result.resid,
                    "observed": result.observed,
                }
            except Exception as e:
                logger.warning(f"STL decomposition failed for {ticker}: {e}")
    
    def classify_trends(self, lookback: int = 60) -> None:
        """Classify trend direction based on slope of trend component.
        
        Args:
            lookback: Number of days to assess trend (default 60)
        """
        for ticker, stl_data in self.stl_results.items():
            trend = stl_data["trend"].dropna()
            
            if len(trend) < lookback:
                continue
            
            # Compute slope over last 'lookback' days
            recent_trend = trend.tail(lookback)
            slope = np.polyfit(range(len(recent_trend)), recent_trend.values, 1)[0]
            
            # Classify based on slope magnitude
            slope_threshold = trend.std() * 0.01  # Relative threshold
            
            if slope > slope_threshold:
                classification = "upward"
            elif slope < -slope_threshold:
                classification = "downward"
            else:
                classification = "sideways"
            
            self.trend_classifications[ticker] = classification
            
            logger.info(f"{ticker} trend: {classification} (slope: {slope:.4f})")
    
    def save_summary(self) -> None:
        """Save volatility and trend summary to CSV."""
        summary_rows = []
        
        for ticker in STOCKS:
            row: Dict[str, object] = {"ticker": ticker}
            
            # Latest volatility values
            if ticker in self.volatility_data:
                for vol_name, vol_series in self.volatility_data[ticker].items():
                    row[vol_name] = vol_series.iloc[-1] if not vol_series.empty else None
            
            # Trend classification
            row["trend_classification"] = self.trend_classifications.get(ticker, "unknown")
            
            # Trend strength (from STL)
            if ticker in self.stl_results:
                resid = self.stl_results[ticker]["resid"]
                trend = self.stl_results[ticker]["trend"]
                if len(resid) > 0 and len(trend) > 0:
                    trend_strength = 1 - (resid.var() / (resid + trend).var())
                    row["trend_strength"] = trend_strength
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        out_path = self.reports_path / "volatility_trend_summary.csv"
        summary_df.to_csv(out_path, index=False)
        
        logger.info(f"Volatility trend summary saved to {out_path}")
