"""Phase 4B: ML/DL Forecasting Models."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from forecasting.prophet_model import ProphetModel
from forecasting.lstm import LSTMModel
from forecasting.gru import GRUModel

print("=" * 60)
print("PHASE 4B: ML/DL FORECASTING")
print("=" * 60)

# Prophet
print("\n[1/3] Training Prophet models...")
prophet = ProphetModel()
prophet.train_all()
print("Backtesting Prophet...")
prophet.backtest_all()
prophet.save_metrics()
print("Prophet complete!")

# LSTM
print("\n[2/3] Training LSTM models...")
lstm = LSTMModel()
lstm.prepare_sequences()
lstm.train_all()
print("Backtesting LSTM...")
lstm.backtest_all()
lstm.save_metrics()
print("LSTM complete!")

# GRU
print("\n[3/3] Training GRU models...")
gru = GRUModel()
gru.prepare_sequences()
gru.train_all()
print("Backtesting GRU...")
gru.backtest_all()
gru.save_metrics()
print("GRU complete!")

print("\n" + "=" * 60)
print("PHASE 4B COMPLETE")
print("=" * 60)
