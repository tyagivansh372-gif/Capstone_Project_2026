"""GARCH(1,1) volatility forecasting module.

Implements conditional volatility forecasting using log returns.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from arch import arch_model

from config.config import STOCKS, PATHS, DATE_RANGES, GARCH_PARAMS
from preprocessing.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class GarchModel:
    """GARCH(1,1) volatility model.
    
    Attributes:
        models: Dictionary of fitted GARCH models per stock
        log_returns: Dictionary of computed log returns
        volatility_forecasts: Dictionary of forecasted volatility
    """
    
    def __init__(self) -> None:
        """Initialize GARCH model container."""
        self.models: Dict[str, arch_model] = {}
        self.log_returns: Dict[str, pd.Series] = {}
        self.volatility_forecasts: Dict[str, pd.DataFrame] = {}
        
        self.forecasts_path: Path = Path(PATHS["outputs_forecasts"])
        self.forecasts_path.mkdir(parents=True, exist_ok=True)
    
    def compute_log_returns(self) -> None:
        """Compute log returns for all stocks from training data."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                log_returns = np.log(prices / prices.shift(1)).dropna()
                self.log_returns[ticker] = log_returns
    
    def fit(self, ticker: str) -> arch_model:
        """Fit GARCH(1,1) model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Fitted ARCH model result
        """
        logger.info(f"Fitting GARCH for {ticker}")
        
        if ticker not in self.log_returns:
            raise ValueError(f"No log returns for {ticker}")
        
        returns = self.log_returns[ticker] * 100  # Scale for numerical stability
        
        model = arch_model(
            returns,
            vol=GARCH_PARAMS["vol"],
            p=GARCH_PARAMS["p"],
            q=GARCH_PARAMS["q"],
            dist=GARCH_PARAMS["dist"],
        )
        
        result = model.fit(disp="off")
        self.models[ticker] = result
        
        logger.info(f"{ticker} GARCH converged: {result.optimization_result.success}")
        
        return result
    
    def fit_all(self) -> None:
        """Fit GARCH models for all stocks."""
        if not self.log_returns:
            self.compute_log_returns()
        
        for ticker in STOCKS:
            if ticker in self.log_returns:
                try:
                    self.fit(ticker)
                except Exception as e:
                    logger.error(f"Failed to fit GARCH for {ticker}: {e}")
    
    def forecast_volatility(self, horizon: int = 5) -> None:
        """Forecast conditional volatility for all stocks.
        
        Args:
            horizon: Number of days to forecast (default 5 for trading week)
        """
        
        start_date = pd.Timestamp(DATE_RANGES["live_forecast_start"])
        
        for ticker in STOCKS:
            if ticker not in self.models:
                continue
            
            result = self.models[ticker]
            forecast = result.forecast(horizon=horizon)
            
            # Extract variance forecasts and convert to volatility
            variance = forecast.variance.values[-1]
            volatility = np.sqrt(variance) / 100  # Undo scaling
            
            # Create forecast dates
            dates = pd.date_range(start=start_date, periods=horizon, freq="B")
            
            self.volatility_forecasts[ticker] = pd.DataFrame({
                "date": dates,
                "forecasted_volatility": volatility,
                "forecasted_variance": variance / 10000,
            })
    
    def save_results(self) -> None:
        """Save volatility forecasts to CSV."""
        if not self.volatility_forecasts:
            return
        
        all_forecasts = []
        for ticker, df in self.volatility_forecasts.items():
            df_copy = df.copy()
            df_copy["stock"] = ticker
            all_forecasts.append(df_copy)
        
        if all_forecasts:
            combined = pd.concat(all_forecasts, ignore_index=True)
            combined = combined[["stock", "date", "forecasted_volatility", "forecasted_variance"]]
            
            out_path = self.forecasts_path / "garch_volatility_may2026.csv"
            combined.to_csv(out_path, index=False)
            
            logger.info(f"GARCH volatility forecasts saved to {out_path}")
