"""Phase 4D: Ensemble Models + Live Forecasts."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from data_fetching.fetcher import DataFetcher
from forecasting.ensemble import EnsembleModel

print("=" * 60)
print("PHASE 4D: ENSEMBLE + LIVE FORECASTS")
print("=" * 60)

# Step 1: Fetch extended data
print("\n[1/5] Fetching extended data up to May 10, 2026...")
fetcher = DataFetcher()
fetcher.fetch_extended_data()
print("Extended data fetched!")

# Step 2: Build ensemble and compute weights
print("\n[2/5] Building ensemble model (ARIMA + Prophet + LSTM)...")
ensemble = EnsembleModel()
ensemble.compute_weights()

# Step 3: Retrain models on extended data
print("\n[3/5] Retraining models on extended dataset...")
ensemble.retrain_all_models()

# Step 4: Generate live forecasts for May 11-15, 2026
print("\n[4/5] Generating live forecasts for May 11-15, 2026...")
ensemble.generate_live_forecasts()

# Step 5: Save results
print("\n[5/5] Saving ensemble forecasts and weights...")
ensemble.save_results()

print("\n" + "=" * 60)
print("PHASE 4D COMPLETE")
print("=" * 60)
