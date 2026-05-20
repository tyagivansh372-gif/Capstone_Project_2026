import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.config import STOCKS, PATHS, PORTFOLIO_CONFIG, STOCK_INFO

print("="*60)
print("RECOMPUTING PORTFOLIO WITH LIVE FORECASTS")
print("="*60)

# Load live forecasts
live_fc = pd.read_csv('outputs/forecasts/live_forecasts_may2026.csv', parse_dates=['date'])

# Load extended data for last prices and volatility
extended = pd.read_csv('data/raw/extended/all_stocks_extended.csv', 
                       header=[0,1], index_col=0, parse_dates=True)

# Load GARCH volatility
vol_df = pd.read_csv('outputs/forecasts/garch_volatility_may2026.csv', parse_dates=['date'])

# Compute predicted returns and get last prices
forecast_returns = {}
last_prices = {}
avg_vols = {}

for ticker in STOCKS:
    # Get last known price from extended data
    try:
        prices = extended[('Close', ticker)].dropna()
        last_prices[ticker] = float(prices.iloc[-1])
    except:
        print(f"Warning: Could not get last price for {ticker}")
        last_prices[ticker] = 1000.0  # fallback
    
    # Compute 5-day predicted return from live forecasts
    stock_fc = live_fc[live_fc['stock'] == ticker].sort_values('date')
    if len(stock_fc) >= 2:
        start_price = stock_fc.iloc[0]['predicted_price']
        end_price = stock_fc.iloc[-1]['predicted_price']
        forecast_returns[ticker] = (end_price / start_price) - 1
    else:
        forecast_returns[ticker] = 0.0
    
    # Get average volatility
    stock_vol = vol_df[vol_df['stock'] == ticker]
    if not stock_vol.empty:
        avg_vols[ticker] = stock_vol['forecasted_volatility'].mean()
    else:
        avg_vols[ticker] = 0.02  # default 2% volatility

print(f"\nForecast returns (5-day):")
for t, r in sorted(forecast_returns.items(), key=lambda x: x[1], reverse=True):
    print(f"  {t}: {r*100:.3f}%")

print(f"\nAverage volatility (5-day forecast):")
for t, v in sorted(avg_vols.items(), key=lambda x: x[1]):
    print(f"  {t}: {v:.4f}")

# Strategy A: Forecast-guided (rank by predicted return)
sorted_stocks = sorted(forecast_returns.items(), key=lambda x: x[1], reverse=True)
n = len(sorted_stocks)
total_rank = sum(range(1, n + 1))
weights_a = {}
for rank, (ticker, ret) in enumerate(sorted_stocks, 1):
    weight = (n - rank + 1) / total_rank
    weights_a[ticker] = weight

# Normalize Strategy A
total_a = sum(weights_a.values())
weights_a = {k: v / total_a for k, v in weights_a.items()}

# Strategy B: Inverse volatility
inv_vols = {t: 1.0 / v for t, v in avg_vols.items() if v > 0}
total_inv = sum(inv_vols.values())
weights_b = {t: v / total_inv for t, v in inv_vols.items()}

print(f"\nStrategy A weights (forecast-guided):")
for t, w in sorted(weights_a.items(), key=lambda x: x[1], reverse=True):
    print(f"  {t}: {w*100:.2f}%")

print(f"\nStrategy B weights (volatility-aware):")
for t, w in sorted(weights_b.items(), key=lambda x: x[1], reverse=True):
    print(f"  {t}: {w*100:.2f}%")

# Combine: 60% A + 40% B (with MARUTI adjustment due to high MAPE)
combined = {}
for ticker in STOCKS:
    # For MARUTI, reduce Strategy A contribution due to high forecast error
    if ticker == 'MARUTI.NS':
        w_a, w_b = 0.40, 0.60  # More weight on volatility, less on forecast
    else:
        w_a, w_b = 0.60, 0.40
    
    combined[ticker] = w_a * weights_a.get(ticker, 0) + w_b * weights_b.get(ticker, 0)

# Normalize
total_combined = sum(combined.values())
combined = {k: v / total_combined for k, v in combined.items()}

# Apply position limits
max_pos = PORTFOLIO_CONFIG["max_position_pct"]
min_pos = PORTFOLIO_CONFIG["min_position_pct"]
combined = {k: max(min(v, max_pos), min_pos) for k, v in combined.items()}

# Renormalize after clipping
total_after_clip = sum(combined.values())
combined = {k: v / total_after_clip for k, v in combined.items()}

# Calculate INR allocation and shares
total_capital = PORTFOLIO_CONFIG["total_capital"]
allocation_records = []

for ticker in STOCKS:
    weight = combined[ticker]
    allocated_inr = weight * total_capital
    last_price = last_prices[ticker]
    shares = int(allocated_inr / last_price)  # floor
    
    # Generate rationale
    ret = forecast_returns[ticker] * 100
    vol = avg_vols[ticker]
    rank = sorted_stocks.index((ticker, forecast_returns[ticker])) + 1
    
    if ticker == 'MARUTI.NS':
        rationale = f"{ticker}: Rank #{rank} with 5-day predicted return of {ret:.2f}% (HIGH MAPE: models unreliable, using 40% Strategy A weight). High volatility score {vol:.4f}"
    else:
        rationale = f"{ticker}: Rank #{rank} with 5-day predicted return of {ret:.2f}% and volatility score of {vol:.4f}"
    
    allocation_records.append({
        'ticker': ticker,
        'weight_pct': round(weight * 100, 2),
        'allocated_INR': round(allocated_inr, 2),
        'shares_to_buy': shares,
        'last_price': round(last_price, 2),
        'predicted_return_5d_pct': round(ret, 3),
        'avg_volatility': round(vol, 4),
        'strategy_rationale': rationale
    })

# Save allocation
allocation_df = pd.DataFrame(allocation_records)
allocation_df.to_csv('outputs/reports/portfolio_allocation.csv', index=False)

print("\n" + "="*60)
print("PORTFOLIO ALLOCATION UPDATED")
print("="*60)
print(allocation_df.to_string(index=False))

total_allocated = allocation_df['allocated_INR'].sum()
print(f"\nTotal allocated: INR {total_allocated:,.2f}")
print(f"Target: INR {total_capital:,.2f}")
print(f"Difference: INR {total_allocated - total_capital:.2f}")
