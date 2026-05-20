"""ARIMA forecasting module.

Auto-tuned ARIMA using pmdarima with AIC/BIC selection and residual validation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox

from config.config import STOCKS, PATHS, ARIMA_PARAMS
from preprocessing.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class ArimaModel:
    """ARIMA time series forecast model.
    
    Attributes:
        models: Dictionary of fitted ARIMA models per stock
        forecasts: Dictionary of forecast results
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self) -> None:
        """Initialize ARIMA model container."""
        self.models: Dict[str, auto_arima] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.models_path: Path = Path(PATHS["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def train(self, ticker: str, train_data: pd.Series) -> auto_arima:
        """Train ARIMA model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            train_data: Training price series
            
        Returns:
            Fitted auto_arima model
        """
        logger.info(f"Training ARIMA for {ticker}")
        
        model = auto_arima(
            train_data,
            **ARIMA_PARAMS,
            trace=False,
        )
        
        # Residual validation with Ljung-Box test
        residuals = model.resid()
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        if lb_test["lb_pvalue"].iloc[0] < 0.05:
            logger.warning(f"{ticker} ARIMA residuals show autocorrelation")
        
        self.models[ticker] = model
        
        # Save model
        model_path = self.models_path / f"arima_{ticker.replace('.', '_')}.pkl"
        joblib.dump(model, model_path)
        
        return model
    
    def train_all(self) -> None:
        """Train ARIMA models for all stocks."""
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
            actual: Actual price series for backtest period
            
        Returns:
            Dictionary of evaluation metrics
        """
        steps = len(actual)
        fc, _ = self.forecast(ticker, steps)
        
        # Compute metrics
        from evaluation.evaluator import ModelEvaluator
        
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
        metrics_df["model"] = "ARIMA"
        
        out_path = Path(PATHS["outputs_metrics"])
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "arima_metrics.csv")
        
        logger.info(f"ARIMA metrics saved to {out_path / 'arima_metrics.csv'}")
