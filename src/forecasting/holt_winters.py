"""Holt-Winters Exponential Smoothing module.

Triple exponential smoothing with trend and optional seasonality.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config.config import STOCKS, PATHS, HOLT_WINTERS_PARAMS
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class HoltWintersModel:
    """Holt-Winters Exponential Smoothing forecast model.
    
    Attributes:
        models: Dictionary of fitted models per stock
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self) -> None:
        """Initialize Holt-Winters model container."""
        self.models: Dict[str, ExponentialSmoothing] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.models_path: Path = Path(PATHS["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def train(self, ticker: str, train_data: pd.Series) -> ExponentialSmoothing:
        """Train Holt-Winters model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            train_data: Training price series
            
        Returns:
            Fitted Holt-Winters model
        """
        logger.info(f"Training Holt-Winters for {ticker}")
        
        # Try both additive and multiplicative, select by AIC
        variants = []
        
        for trend in ["add", "mul"]:
            try:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=None,  # No seasonality for daily stock prices
                    initialization_method=HOLT_WINTERS_PARAMS["initialization_method"],
                ).fit()
                variants.append((model, trend, model.aic))
            except Exception as e:
                logger.warning(f"Holt-Winters {trend} failed for {ticker}: {e}")
        
        if not variants:
            raise ValueError(f"No valid Holt-Winters model for {ticker}")
        
        # Select best by AIC
        best_model, best_trend, best_aic = min(variants, key=lambda x: x[2])
        
        logger.info(f"{ticker} selected {best_trend} trend (AIC: {best_aic:.2f})")
        
        self.models[ticker] = best_model
        
        # Save model
        model_path = self.models_path / f"holt_winters_{ticker.replace('.', '_')}.pkl"
        joblib.dump(best_model, model_path)
        
        return best_model
    
    def train_all(self) -> None:
        """Train Holt-Winters models for all stocks."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                try:
                    self.train(ticker, prices)
                except Exception as e:
                    logger.error(f"Failed to train Holt-Winters for {ticker}: {e}")
    
    def forecast(self, ticker: str, steps: int) -> np.ndarray:
        """Generate forecast for a stock.
        
        Args:
            ticker: Stock ticker
            steps: Number of steps to forecast
            
        Returns:
            Array of point forecasts
        """
        if ticker not in self.models:
            raise ValueError(f"Model not trained for {ticker}")
        
        model = self.models[ticker]
        fc = model.forecast(steps)
        
        return fc.values
    
    def backtest(self, ticker: str, actual: pd.Series) -> Dict[str, float]:
        """Backtest model on holdout period.
        
        Args:
            ticker: Stock ticker
            actual: Actual price series
            
        Returns:
            Dictionary of evaluation metrics
        """
        steps = len(actual)
        fc = self.forecast(ticker, steps)
        
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
        metrics_df["model"] = "Holt-Winters"
        
        out_path = Path(PATHS["outputs_metrics"])
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "holt_winters_metrics.csv")
        
        logger.info(f"Holt-Winters metrics saved to {out_path / 'holt_winters_metrics.csv'}")
