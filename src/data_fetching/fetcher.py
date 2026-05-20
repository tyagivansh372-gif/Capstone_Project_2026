"""Data fetching module for NSE stocks via yfinance.

Downloads historical price data and performs initial validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config.config import STOCKS, DATE_RANGES, PATHS

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and validates historical stock data from Yahoo Finance.
    
    Attributes:
        stocks: List of ticker symbols to fetch
        date_ranges: Dictionary of start/end dates
        raw_path: Path to save raw data
    """
    
    def __init__(self) -> None:
        """Initialize the data fetcher with configuration."""
        self.stocks: List[str] = STOCKS
        self.date_ranges: Dict[str, str] = DATE_RANGES
        self.raw_path: Path = Path(PATHS["data_raw"])
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, pd.DataFrame] = {}
        
    def fetch_stock(
        self, 
        ticker: str, 
        start: str, 
        end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        logger.info(f"Fetching {ticker} from {start} to {end}")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None
    
    def fetch_all(self) -> None:
        """Fetch data for all configured stocks (base period)."""
        start = self.date_ranges["train_start"]
        end = self.date_ranges["backtest_end"]
        
        for ticker in self.stocks:
            df = self.fetch_stock(ticker, start, end)
            if df is not None:
                # Flatten MultiIndex columns in the stored data
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Ensure index is datetime
                df.index = pd.to_datetime(df.index)
                self.data[ticker] = df
                self._save_raw(ticker, df)
    
    def fetch_extended_data(self) -> None:
        """Fetch extended dataset up to live forecast date."""
        start = self.date_ranges["train_start"]
        end = self.date_ranges["extended_train_end"]
        
        extended_path = self.raw_path / "extended"
        extended_path.mkdir(exist_ok=True)
        
        for ticker in self.stocks:
            df = self.fetch_stock(ticker, start, end)
            if df is not None:
                # Validate freshness
                if not self._validate_freshness(df, end):
                    raise ValueError(
                        f"Data for {ticker} is stale. Latest date: {df.index[-1]}"
                    )
                self._save_to_path(ticker, df, extended_path)
    
    def _save_raw(self, ticker: str, df: pd.DataFrame) -> None:
        """Save raw data to CSV with flattened columns."""
        filepath = self.raw_path / f"{ticker.replace('.', '_')}.csv"
        
        # Work on a copy to avoid modifying original
        df_out = df.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(df_out.columns, pd.MultiIndex):
            df_out.columns = df_out.columns.get_level_values(0)
        
        # Ensure index is a simple DatetimeIndex with name 'Date'
        df_out.index = pd.to_datetime(df_out.index)
        df_out.index.name = 'Date'
        
        # Ensure Close is numeric
        df_out["Close"] = pd.to_numeric(df_out["Close"], errors="coerce")
        
        # Save with index (Date column)
        df_out.to_csv(filepath)
        logger.info(f"Saved {ticker} to {filepath}")
    
    def _save_to_path(
        self, 
        ticker: str, 
        df: pd.DataFrame, 
        path: Path
    ) -> None:
        """Save data to specified path with flattened columns."""
        filepath = path / f"{ticker.replace('.', '_')}.csv"
        
        # Work on a copy
        df_out = df.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(df_out.columns, pd.MultiIndex):
            df_out.columns = df_out.columns.get_level_values(0)
        
        # Ensure index is simple DatetimeIndex
        df_out.index = pd.to_datetime(df_out.index)
        df_out.index.name = 'Date'
        
        # Ensure Close is numeric
        df_out["Close"] = pd.to_numeric(df_out["Close"], errors="coerce")
        
        df_out.to_csv(filepath)
        logger.info(f"Saved {ticker} to {filepath}")
    
    def _validate_freshness(
        self, 
        df: pd.DataFrame, 
        expected_end: str
    ) -> bool:
        """Validate that data is fresh (within 3 days of expected end)."""
        
        latest = df.index[-1]
        expected = pd.Timestamp(expected_end)
        
        # Allow 3 calendar days tolerance
        delta = abs((latest - expected).days)
        return delta <= 3
    
    def validate_data(self) -> Dict[str, Dict[str, int]]:
        """Validate fetched data for quality issues.
        
        Returns:
            Dictionary of validation results per stock
        """
        results: Dict[str, Dict[str, int]] = {}
        
        for ticker, df in self.data.items():
            issues: Dict[str, int] = {
                "missing_dates": 0,
                "zero_volume": 0,
                "nan_count": 0,
            }
            
            # Check for NaN values
            issues["nan_count"] = int(df.isna().sum().sum())
            
            # Check for zero volume
            if "Volume" in df.columns:
                issues["zero_volume"] = int((df["Volume"] == 0).sum())
            
            # Check for missing trading dates
            all_dates = pd.date_range(df.index[0], df.index[-1], freq="B")
            missing = len(all_dates) - len(df)
            issues["missing_dates"] = max(0, missing)
            
            results[ticker] = issues
            
            if any(v > 0 for v in issues.values()):
                logger.warning(f"{ticker} validation issues: {issues}")
        
        return results
    
    def generate_selection_report(self) -> None:
        """Generate stock selection justification report.
        
        Uses rolling std dev, STL trend strength, and momentum analysis.
        """
        from statsmodels.tsa.seasonal import STL
        
        report_rows: List[Dict[str, float]] = []
        
        for ticker, df in self.data.items():
            prices = df["Close"].squeeze()
            
            # Rolling 30-day volatility (std dev)
            rolling_vol = prices.pct_change().rolling(30).std().mean()
            
            # STL decomposition trend strength
            stl = STL(prices, period=5, robust=True)
            result = stl.fit()
            trend_strength = 1 - (result.resid.var() / (result.resid + result.trend).var())
            
            # 6-month momentum
            momentum = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) > 126 else 0
            
            report_rows.append({
                "ticker": ticker,
                "rolling_vol_30d": rolling_vol,
                "trend_strength": trend_strength,
                "momentum_6m": momentum,
            })
        
        report_df = pd.DataFrame(report_rows)
        
        out_path = Path(PATHS["outputs_reports"])
        out_path.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path / "stock_selection.csv", index=False)
        
        logger.info(f"Stock selection report saved to {out_path / 'stock_selection.csv'}")
