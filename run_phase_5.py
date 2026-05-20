"""Phase 5: Volatility and Trend Analysis."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from volatility.analyzer import VolatilityAnalyzer

print("=" * 60)
print("PHASE 5: VOLATILITY & TREND ANALYSIS")
print("=" * 60)

print("\n[1/3] Computing rolling volatility...")
va = VolatilityAnalyzer()
va.compute_rolling_volatility()

print("\n[2/3] Performing STL decomposition and trend classification...")
va.stl_decomposition()
va.classify_trends()

print("\n[3/3] Saving analysis summary...")
va.save_summary()

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
