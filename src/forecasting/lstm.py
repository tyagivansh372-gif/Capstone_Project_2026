"""LSTM (Long Short-Term Memory) forecasting module.

Deep learning model with sequence windowing, 2 LSTM layers, dropout 0.2.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from config.config import STOCKS, PATHS, DEEP_LEARNING_PARAMS, RANDOM_SEED
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class LSTMModel:
    """LSTM neural network forecast model.
    
    Attributes:
        models: Dictionary of trained LSTM models per stock
        sequences: Dictionary of prepared sequence data
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self) -> None:
        """Initialize LSTM model container."""
        self.models: Dict[str, Sequential] = {}
        self.sequences: Dict[str, Dict[str, np.ndarray]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.models_path: Path = Path(PATHS["models"])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.seq_length: int = DEEP_LEARNING_PARAMS["sequence_length"]
        self.lstm_units: List[int] = DEEP_LEARNING_PARAMS["lstm_units"]
        self.dropout: float = DEEP_LEARNING_PARAMS["dropout"]
        self.lr: float = DEEP_LEARNING_PARAMS["learning_rate"]
        self.batch_size: int = DEEP_LEARNING_PARAMS["batch_size"]
        self.epochs: int = DEEP_LEARNING_PARAMS["epochs"]
        self.patience: int = DEEP_LEARNING_PARAMS["early_stopping_patience"]
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.
        
        Args:
            data: Scaled price array
            seq_length: Length of input sequences
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def prepare_sequences(self) -> None:
        """Prepare sequence data for all stocks using preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.apply_scaling()
        
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                train_scaled = preprocessor.get_scaled_data(ticker, "train")
                
                if train_scaled is None:
                    continue
                
                # Create sequences
                X_train, y_train = self._create_sequences(train_scaled, self.seq_length)
                
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                
                self.sequences[ticker] = {
                    "X_train": X_train,
                    "y_train": y_train,
                    "scaler": preprocessor.scalers.get(ticker),
                }
    
    def _build_model(self) -> Sequential:
        """Build LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(
                self.lstm_units[0],
                return_sequences=True,
                input_shape=(self.seq_length, 1)
            ),
            Dropout(self.dropout),
            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        
        return model
    
    def train(self, ticker: str) -> Sequential:
        """Train LSTM model for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Trained Keras model
        """
        logger.info(f"Training LSTM for {ticker}")
        
        if ticker not in self.sequences:
            raise ValueError(f"No sequence data for {ticker}")
        
        X_train = self.sequences[ticker]["X_train"]
        y_train = self.sequences[ticker]["y_train"]
        
        model = self._build_model()
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=0,
        )
        
        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=DEEP_LEARNING_PARAMS["validation_split"],
            callbacks=[early_stop],
            verbose=0,
        )
        
        self.models[ticker] = model
        
        # Save model
        model_path = self.models_path / f"lstm_{ticker.replace('.', '_')}.keras"
        model.save(model_path)
        
        logger.info(f"{ticker} LSTM trained for {len(history.history['loss'])} epochs")
        
        return model
    
    def train_all(self) -> None:
        """Train LSTM models for all stocks."""
        if not self.sequences:
            self.prepare_sequences()
        
        for ticker in STOCKS:
            if ticker in self.sequences:
                try:
                    self.train(ticker)
                except Exception as e:
                    logger.error(f"Failed to train LSTM for {ticker}: {e}")
    
    def forecast(self, ticker: str, steps: int) -> np.ndarray:
        """Generate forecast for a stock.
        
        Args:
            ticker: Stock ticker
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasts (original scale)
        """
        if ticker not in self.models or ticker not in self.sequences:
            raise ValueError(f"Model not trained for {ticker}")
        
        model = self.models[ticker]
        scaler = self.sequences[ticker]["scaler"]
        
        # Get last sequence from training data
        last_sequence = self.sequences[ticker]["X_train"][-1].reshape(1, self.seq_length, 1)
        
        forecasts = []
        current_seq = last_sequence.copy()
        
        for _ in range(steps):
            pred = model.predict(current_seq, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence (rolling window)
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred
        
        # Inverse transform to original scale
        forecasts = np.array(forecasts).reshape(-1, 1)
        if scaler:
            forecasts = scaler.inverse_transform(forecasts).flatten()
        
        return forecasts
    
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
        metrics_df["model"] = "LSTM"
        
        out_path = Path(PATHS["outputs_metrics"])
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "lstm_metrics.csv")
        
        logger.info(f"LSTM metrics saved to {out_path / 'lstm_metrics.csv'}")
