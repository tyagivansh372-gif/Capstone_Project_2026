"""Phase 4A: Statistical Forecasting Models."""
import sys
sys.path.insert(0, 'src')

from forecasting.arima import ArimaModel
from forecasting.sarima import SarimaModel
from forecasting.holt_winters import HoltWintersModel

print("=" * 60)
print("PHASE 4A: STATISTICAL FORECASTING")
print("=" * 60)

# ARIMA
print("\n[1/3] Training ARIMA models...")
arima = ArimaModel()
arima.train_all()
print("Backtesting ARIMA...")
arima.backtest_all()
arima.save_metrics()
print("ARIMA complete!")

# SARIMA
print("\n[2/3] Training SARIMA models...")
sarima = SarimaModel()
sarima.train_all()
print("Backtesting SARIMA...")
sarima.backtest_all()
sarima.save_metrics()
print("SARIMA complete!")

# Holt-Winters
print("\n[3/3] Training Holt-Winters models...")
hw = HoltWintersModel()
hw.train_all()
print("Backtesting Holt-Winters...")
hw.backtest_all()
hw.save_metrics()
print("Holt-Winters complete!")

print("\n" + "=" * 60)
print("PHASE 4A COMPLETE")
print("=" * 60)
