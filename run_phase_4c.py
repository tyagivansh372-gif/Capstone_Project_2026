"""Phase 4C: GARCH Volatility Modeling."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from forecasting.garch import GarchModel

print("=" * 60)
print("PHASE 4C: GARCH VOLATILITY MODELING")
print("=" * 60)

print("\n[1/4] Computing log returns...")
garch = GarchModel()
garch.compute_log_returns()

print("\n[2/4] Fitting GARCH(1,1) models...")
garch.fit_all()

print("\n[3/4] Forecasting volatility for May 11-15, 2026...")
garch.forecast_volatility()

print("\n[4/4] Saving volatility results...")
garch.save_results()

print("\n" + "=" * 60)
print("PHASE 4C COMPLETE")
print("=" * 60)
