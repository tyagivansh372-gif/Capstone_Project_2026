import os
import pandas as pd
import sys
sys.path.insert(0, 'src')

print('='*70)
print('FINAL PRE-PHASE 7 VERIFICATION')
print('='*70)

blockers = []
warnings_list = []

# 1. Critical files exist
print('\n[1] CRITICAL FILES CHECK')
required = {
    'outputs/forecasts/live_forecasts_may2026.csv': 16,
    'outputs/forecasts/garch_volatility_may2026.csv': 40,
    'outputs/reports/portfolio_allocation.csv': 8,
    'data/external/live_actuals_may2026.csv': 16,
    '.gitignore': None
}
for path, expected in required.items():
    if not os.path.exists(path):
        blockers.append('MISSING: ' + path)
    elif expected:
        df = pd.read_csv(path)
        if len(df) != expected:
            blockers.append('WRONG ROWS: ' + path + ' (' + str(len(df)) + '/' + str(expected) + ')')
        else:
            print('  PASS: ' + path + ' (' + str(len(df)) + ' rows)')
    else:
        print('  PASS: ' + path)

# 2. No 0% directional accuracy
print('\n[2] DIRECTIONAL ACCURACY CHECK (no 0% allowed)')
mc = pd.read_csv('outputs/metrics/model_comparison.csv')
zero_da = mc[mc['directional_accuracy'] == 0]
if len(zero_da) > 0:
    for _, row in zero_da.iterrows():
        blockers.append('0% DA: ' + row['ticker'] + ' ' + row['model'])
else:
    print('  PASS: No models with 0% directional accuracy')

# 3. No MAPE < 1%
print('\n[3] MAPE SANITY CHECK (no < 1% allowed)')
low_mape = mc[mc['mape'] < 1]
if len(low_mape) > 0:
    for _, row in low_mape.iterrows():
        blockers.append('Suspicious MAPE: ' + row['ticker'] + ' ' + row['model'] + ' = ' + str(round(row['mape'], 2)) + '%')
else:
    print('  PASS: No models with MAPE < 1%')

# 4. Portfolio allocation check
print('\n[4] PORTFOLIO VALIDATION')
port = pd.read_csv('outputs/reports/portfolio_allocation.csv')
total = port['allocated_INR'].sum()
if abs(total - 1000000) > 1:
    blockers.append('Portfolio total = Rs.' + str(round(total, 2)) + ' (expected 10,00,000)')
else:
    print('  PASS: Total allocation = Rs.' + str(round(total, 2)))

if len(port) != 8:
    blockers.append('Portfolio has ' + str(len(port)) + ' stocks (expected 8)')
else:
    print('  PASS: All 8 stocks allocated')

max_weight = port['weight_pct'].max()
if max_weight > 40:
    warnings_list.append('Max weight = ' + str(round(max_weight, 1)) + '% (> 40%)')
else:
    print('  PASS: Max weight = ' + str(round(max_weight, 1)) + '% (<= 40%)')

# 5. Live forecasts sanity
print('\n[5] LIVE FORECASTS SANITY')
live = pd.read_csv('outputs/forecasts/live_forecasts_may2026.csv')
nulls = live.isnull().sum().sum()
if nulls > 0:
    blockers.append('Live forecasts have ' + str(nulls) + ' null values')
else:
    print('  PASS: No null values in live forecasts')

# Summary
print('\n' + '='*70)
if blockers:
    print('BLOCKERS - DO NOT PROCEED TO PHASE 7:')
    for b in blockers:
        print('  ERROR: ' + b)
    print('CLEARED: NO')
else:
    print('ALL CHECKS PASSED')
    if warnings_list:
        print('WARNINGS (non-blocking):')
        for w in warnings_list:
            print('  ' + w)
    print('CLEARED: YES - PROCEED TO PHASE 7')
print('='*70)
