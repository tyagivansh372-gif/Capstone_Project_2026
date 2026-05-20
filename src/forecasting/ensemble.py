"""Ensemble forecasting module.

Weighted average ensemble of ARIMA + Prophet + LSTM.
Weights derived from inverse-MAPE on backtest set.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from config.config import STOCKS, PATHS, DATE_RANGES
from forecasting.arima import ArimaModel
from forecasting.prophet_model import ProphetModel
from forecasting.lstm import LSTMModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble forecast model.
    
    Combines ARIMA, Prophet, and LSTM using inverse-MAPE weights.
    
    Attributes:
        weights: Dictionary of model weights per stock
        forecasts: Dictionary of ensemble forecasts
        model_instances: Dictionary of model class instances
    """
    
    def __init__(self) -> None:
        """Initialize ensemble container."""
        self.weights: Dict[str, Dict[str, float]] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.model_instances: Dict[str, object] = {}
        
        self.metrics_path: Path = Path(PATHS["outputs_metrics"])
        self.forecasts_path: Path = Path(PATHS["outputs_forecasts"])
        
        self.model_classes = {
            "ARIMA": ArimaModel,
            "Prophet": ProphetModel,
            "LSTM": LSTMModel,
        }
    
    def compute_weights(self) -> None:
        """Compute ensemble weights from inverse-MAPE on backtest results."""
        # Load backtest metrics
        metrics_files = {
            "ARIMA": "arima_metrics.csv",
            "Prophet": "prophet_metrics.csv",
            "LSTM": "lstm_metrics.csv",
        }
        
        mape_data: Dict[str, Dict[str, float]] = {m: {} for m in self.model_classes}
        
        for model_name, filename in metrics_files.items():
            filepath = self.metrics_path / filename
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0)
                for ticker in df.index:
                    if "mape" in df.columns:
                        mape_data[model_name][ticker] = df.loc[ticker, "mape"]
                    elif "MAPE" in df.columns:
                        mape_data[model_name][ticker] = df.loc[ticker, "MAPE"]
        
        # Compute inverse-MAPE weights
        for ticker in STOCKS:
            inv_mape: Dict[str, float] = {}
            
            for model_name in self.model_classes:
                if ticker in mape_data[model_name] and mape_data[model_name][ticker] > 0:
                    inv_mape[model_name] = 1.0 / mape_data[model_name][ticker]
            
            # Normalize to sum to 1
            total = sum(inv_mape.values())
            if total > 0:
                self.weights[ticker] = {
                    m: w / total for m, w in inv_mape.items()
                }
            else:
                # Equal weights if no metrics
                n_models = len(self.model_classes)
                self.weights[ticker] = {m: 1.0 / n_models for m in self.model_classes}
        
        # Save weights
        weights_df = pd.DataFrame(self.weights).T
        weights_df.index.name = "ticker"
        weights_df.to_csv(self.metrics_path / "ensemble_weights.csv")
        
        logger.info("Ensemble weights computed and saved")
    
    def retrain_all_models(self) -> None:
        """Retrain all models on extended dataset (Jan 2021 → May 10, 2026)."""
        logger.info("Retraining all models on extended dataset")
        
        # This will be implemented to load extended data and retrain
        # For now, instantiate models
        for name, model_class in self.model_classes.items():
            self.model_instances[name] = model_class()
        
        # Load extended data and retrain (implementation detail)
        # Each model's retrain logic will be called here
    
    def generate_live_forecasts(self, steps: int = 5) -> None:
        """Generate ensemble forecasts for May 11-15, 2026.
        
        Args:
            steps: Number of trading days to forecast (default 5)
        """
        
        start_date = pd.Timestamp(DATE_RANGES["live_forecast_start"])
        dates = pd.date_range(start=start_date, periods=steps, freq="B")
        
        all_forecasts = []
        
        for ticker in STOCKS:
            if ticker not in self.weights:
                continue
            
            # Collect forecasts from each model
            model_forecasts: Dict[str, np.ndarray] = {}
            conf_intervals: Optional[np.ndarray] = None
            
            # ARIMA
            if "ARIMA" in self.model_instances and ticker in self.weights[ticker]:
                try:
                    arima = self.model_instances["ARIMA"]
                    fc, conf = arima.forecast(ticker, steps)
                    model_forecasts["ARIMA"] = fc
                    conf_intervals = conf
                except Exception as e:
                    logger.warning(f"ARIMA forecast failed for {ticker}: {e}")
            
            # Prophet
            if "Prophet" in self.model_instances and ticker in self.weights[ticker]:
                try:
                    prophet = self.model_instances["Prophet"]
                    fc, lower, upper = prophet.forecast(ticker, steps)
                    model_forecasts["Prophet"] = fc
                    conf_intervals = np.column_stack([lower, upper])
                except Exception as e:
                    logger.warning(f"Prophet forecast failed for {ticker}: {e}")
            
            # LSTM
            if "LSTM" in self.model_instances and ticker in self.weights[ticker]:
                try:
                    lstm = self.model_instances["LSTM"]
                    fc = lstm.forecast(ticker, steps)
                    model_forecasts["LSTM"] = fc
                except Exception as e:
                    logger.warning(f"LSTM forecast failed for {ticker}: {e}")
            
            # Compute weighted ensemble
            if model_forecasts:
                ensemble_fc = np.zeros(steps)
                for model_name, fc in model_forecasts.items():
                    weight = self.weights[ticker].get(model_name, 0)
                    ensemble_fc += weight * fc
                
                # Create forecast records
                for i, date in enumerate(dates):
                    all_forecasts.append({
                        "stock": ticker,
                        "date": date,
                        "predicted_price": ensemble_fc[i],
                        "model": "ensemble",
                        "confidence_interval_low": conf_intervals[i, 0] if conf_intervals is not None else None,
                        "confidence_interval_high": conf_intervals[i, 1] if conf_intervals is not None else None,
                    })
                
                # Add individual model forecasts
                for model_name, fc in model_forecasts.items():
                    for i, date in enumerate(dates):
                        all_forecasts.append({
                            "stock": ticker,
                            "date": date,
                            "predicted_price": fc[i],
                            "model": model_name.lower(),
                            "confidence_interval_low": None,
                            "confidence_interval_high": None,
                        })
        
        # Save forecasts
        if all_forecasts:
            forecasts_df = pd.DataFrame(all_forecasts)
            forecasts_df.to_csv(
                self.forecasts_path / "live_forecasts_may2026.csv",
                index=False
            )
            logger.info(f"Live forecasts saved with {len(forecasts_df)} rows")
    
    def save_results(self) -> None:
        """Save ensemble results (weights already saved in compute_weights)."""
        # Additional saving if needed
