"""SARIMA forecasting module.

Seasonal ARIMA with weekly trading seasonality (m=5).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from pmdarima import auto_arima

from config.config import STOCKS, PATHS, SARIMA_PARAMS
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class SarimaModel:
    """SARIMA time series forecast model with weekly seasonality.
    
    Attributes:
        models: Dictionary of fitted SARIMA models per stock
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self) -> None:
        """Initialize SARIMA model container."""
        self.models: Dict[str, auto_arima] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.models_path: Path = Path(PATHS["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def train(self, ticker: str, train_data: pd.Series) -> auto_arima:
        """Train SARIMA model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            train_data: Training price series
            
        Returns:
            Fitted SARIMA model
        """
        logger.info(f"Training SARIMA for {ticker}")
        
        model = auto_arima(
            train_data,
            **SARIMA_PARAMS,
            trace=False,
        )
        
        self.models[ticker] = model
        
        # Save model
        model_path = self.models_path / f"sarima_{ticker.replace('.', '_')}.pkl"
        joblib.dump(model, model_path)
        
        return model
    
    def train_all(self) -> None:
        """Train SARIMA models for all stocks."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                self.train(ticker, prices)
    
    def forecast(
        self, 
        ticker: str, 
        steps: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate forecast for a stock.
        
        Args:
            ticker: Stock ticker
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (point forecasts, confidence intervals)
        """
        if ticker not in self.models:
            raise ValueError(f"Model not trained for {ticker}")
        
        model = self.models[ticker]
        fc, conf_int = model.predict(n_periods=steps, return_conf_int=True)
        
        return fc, conf_int
    
    def backtest(self, ticker: str, actual: pd.Series) -> Dict[str, float]:
        """Backtest model on holdout period.
        
        Args:
            ticker: Stock ticker
            actual: Actual price series
            
        Returns:
            Dictionary of evaluation metrics
        """
        steps = len(actual)
        fc, _ = self.forecast(ticker, steps)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.compute_single_metrics(actual.values, fc)
        
        self.metrics[ticker] = metrics
        return metrics
    
    def backtest_all(self) -> None:
        """Backtest all models on the backtest period."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker in preprocessor.backtest_data and ticker in self.models:
                actual = preprocessor.backtest_data[ticker]["Close"].squeeze()
                self.backtest(ticker, actual)
    
    def save_metrics(self) -> None:
        """Save evaluation metrics to CSV."""
        if not self.metrics:
            return
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = "ticker"
        metrics_df["model"] = "SARIMA"
        
        out_path = Path(PATHS["outputs_metrics"])
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "sarima_metrics.csv")
        
        logger.info(f"SARIMA metrics saved to {out_path / 'sarima_metrics.csv'}")
