"""Phase 6: Portfolio Optimization."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from portfolio.optimizer import PortfolioOptimizer

print("=" * 60)
print("PHASE 6: PORTFOLIO OPTIMIZATION")
print("=" * 60)

print("\n[1/4] Loading forecast and volatility data...")
po = PortfolioOptimizer()
po.load_forecasts()

print("\n[2/4] Running allocation strategies...")
po.strategy_a_forecast_guided()
po.strategy_b_volatility_aware()
po.strategy_c_correlation_based()
po.strategy_d_sector_momentum()

print("\n[3/4] Combining strategies (60% forecast + 40% volatility)...")
po.combine_strategies()

print("\n[4/4] Saving portfolio allocation...")
po.save_allocation()

print("\n" + "=" * 60)
print("PHASE 6 COMPLETE")
print("=" * 60)
