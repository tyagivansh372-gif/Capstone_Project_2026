"""Facebook Prophet forecasting module.

Implements Prophet with Indian market holidays and confidence intervals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import holidays

from config.config import STOCKS, PATHS, PROPHET_PARAMS
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ProphetModel:
    """Facebook Prophet forecast model with Indian holidays.
    
    Attributes:
        models: Dictionary of fitted Prophet models per stock
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self) -> None:
        """Initialize Prophet model container."""
        self.models: Dict[str, Prophet] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.models_path: Path = Path(PATHS["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def _get_indian_holidays(self, years: List[int]) -> pd.DataFrame:
        """Generate Indian holiday dataframe for Prophet.
        
        Args:
            years: List of years to include
            
        Returns:
            DataFrame with holiday dates and names
        """
        india_holidays = holidays.India(years=years)
        
        holiday_df = pd.DataFrame([
            {"holiday": name, "ds": date}
            for date, name in india_holidays.items()
        ])
        
        return holiday_df
    
    def train(self, ticker: str, train_data: pd.Series) -> Prophet:
        """Train Prophet model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            train_data: Training price series
            
        Returns:
            Fitted Prophet model
        """
        logger.info(f"Training Prophet for {ticker}")
        
        # Prepare data in Prophet format
        df = pd.DataFrame({
            "ds": train_data.index,
            "y": train_data.values,
        })
        
        # Get years for holidays
        years = list(range(df["ds"].min().year, df["ds"].max().year + 1))
        holidays_df = self._get_indian_holidays(years)
        
        # Create and fit model
        model = Prophet(
            yearly_seasonality=PROPHET_PARAMS["yearly_seasonality"],
            weekly_seasonality=PROPHET_PARAMS["weekly_seasonality"],
            daily_seasonality=PROPHET_PARAMS["daily_seasonality"],
            interval_width=PROPHET_PARAMS["interval_width"],
            changepoint_prior_scale=PROPHET_PARAMS["changepoint_prior_scale"],
            seasonality_prior_scale=PROPHET_PARAMS["seasonality_prior_scale"],
        )
        
        if not holidays_df.empty:
            model = Prophet(
                holidays=holidays_df,
                yearly_seasonality=PROPHET_PARAMS["yearly_seasonality"],
                weekly_seasonality=PROPHET_PARAMS["weekly_seasonality"],
                daily_seasonality=PROPHET_PARAMS["daily_seasonality"],
                interval_width=PROPHET_PARAMS["interval_width"],
            )
        
        model.fit(df)
        self.models[ticker] = model
        
        # Save model
        model_path = self.models_path / f"prophet_{ticker.replace('.', '_')}.pkl"
        joblib.dump(model, model_path)
        
        return model
    
    def train_all(self) -> None:
        """Train Prophet models for all stocks."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                self.train(ticker, prices)
    
    def forecast(
        self, 
        ticker: str, 
        periods: int,
        freq: str = "B"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast for a stock.
        
        Args:
            ticker: Stock ticker
            periods: Number of periods to forecast
            freq: Frequency string (default 'B' for business days)
            
        Returns:
            Tuple of (forecasts, lower bound, upper bound)
        """
        if ticker not in self.models:
            raise ValueError(f"Model not trained for {ticker}")
        
        model = self.models[ticker]
        
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        
        # Get last 'periods' rows for predictions
        pred = forecast.tail(periods)
        
        return (
            pred["yhat"].values,
            pred["yhat_lower"].values,
            pred["yhat_upper"].values,
        )
    
    def backtest(self, ticker: str, actual: pd.Series) -> Dict[str, float]:
        """Backtest model on holdout period.
        
        Args:
            ticker: Stock ticker
            actual: Actual price series
            
        Returns:
            Dictionary of evaluation metrics
        """
        periods = len(actual)
        fc, _, _ = self.forecast(ticker, periods)
        
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
        metrics_df["model"] = "Prophet"
        
        out_path = Path(PATHS["outputs_metrics"])
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "prophet_metrics.csv")
        
        logger.info(f"Prophet metrics saved to {out_path / 'prophet_metrics.csv'}")
