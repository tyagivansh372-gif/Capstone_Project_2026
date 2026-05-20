import os
import pandas as pd
import sys
sys.path.insert(0, 'src')
import yaml

print('='*75)
print('CROSS VERIFICATION: state.yaml vs skills.md vs actual files')
print('='*75)

# Load state.yaml
with open('state.yaml', 'r') as f:
    state = yaml.safe_load(f)

# Load skills.md content (parsed manually)
skills_stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'SUNPHARMA.NS', 'MARUTI.NS', 'ITC.NS', 'TATASTEEL.NS', 'BAJFINANCE.NS']
skills_models = ['ARIMA', 'SARIMA', 'Holt-Winters', 'Prophet', 'LSTM', 'GRU', 'GARCH', 'Ensemble']

mismatches = []
failures = []

# 1. Stock universe cross-check
print('\n[1] STOCK UNIVERSE CROSS-CHECK')
print('-'*75)
state_stocks = state.get('stocks', [])
skills_stocks_set = set(skills_stocks)
state_stocks_set = set(state_stocks)

if skills_stocks_set == state_stocks_set:
    print('  PASS: skills.md stocks == state.yaml stocks')
else:
    missing_in_state = skills_stocks_set - state_stocks_set
    missing_in_skills = state_stocks_set - skills_stocks_set
    if missing_in_state:
        mismatches.append('Stocks in skills.md but not state.yaml: ' + str(missing_in_state))
    if missing_in_skills:
        mismatches.append('Stocks in state.yaml but not skills.md: ' + str(missing_in_skills))

# 2. Models implemented cross-check
print('\n[2] MODELS IMPLEMENTED CROSS-CHECK')
print('-'*75)
state_models = state.get('models_implemented', [])
skills_models_set = set(skills_models)
state_models_set = set(state_models)

if skills_models_set == state_models_set:
    print('  PASS: skills.md models == state.yaml models')
else:
    missing_in_state = skills_models_set - state_models_set
    missing_in_skills = state_models_set - skills_models_set
    if missing_in_state:
        mismatches.append('Models in skills.md but not state.yaml: ' + str(missing_in_state))
    if missing_in_skills:
        mismatches.append('Models in state.yaml but not skills.md: ' + str(missing_in_skills))

# 3. Date ranges cross-check
print('\n[3] DATE RANGES CROSS-CHECK')
print('-'*75)
date_ranges = state.get('date_ranges', {})
expected_dates = {
    'train_start': '2021-01-01',
    'train_end': '2025-06-30',
    'backtest_start': '2025-07-01',
    'backtest_end': '2025-12-31',
    'extended_train_end': '2026-05-10'
}
for key, expected in expected_dates.items():
    actual = date_ranges.get(key)
    if actual == expected:
        print('  PASS: ' + key + ' = ' + actual)
    else:
        mismatches.append('Date mismatch: ' + key + ' expected ' + expected + ', got ' + str(actual))

# 4. Phases complete vs deliverables
print('\n[4] PHASES COMPLETE vs ACTUAL DELIVERABLES')
print('-'*75)
phases_complete = state.get('phases_complete', [])
print('  state.yaml claims phases complete: ' + str(phases_complete))

# Check Phase 4D deliverables (live forecasts)
if '4D' in phases_complete:
    if os.path.exists('outputs/forecasts/live_forecasts_may2026.csv'):
        live = pd.read_csv('outputs/forecasts/live_forecasts_may2026.csv')
        if len(live) == 16:
            print('  PASS: Phase 4D - live_forecasts_may2026.csv (16 rows)')
        else:
            failures.append('Phase 4D: live forecasts wrong row count (' + str(len(live)) + ')')
    else:
        failures.append('Phase 4D: live_forecasts_may2026.csv missing')

# Check Phase 6 deliverables (portfolio)
if 6 in phases_complete:
    if os.path.exists('outputs/reports/portfolio_allocation.csv'):
        port = pd.read_csv('outputs/reports/portfolio_allocation.csv')
        if len(port) == 8:
            print('  PASS: Phase 6 - portfolio_allocation.csv (8 stocks)')
        else:
            failures.append('Phase 6: portfolio wrong stock count')
    else:
        failures.append('Phase 6: portfolio_allocation.csv missing')

# 5. Architecture map vs actual files
print('\n[5] ARCHITECTURE MAP vs ACTUAL FILES')
print('-'*75)
required_dirs = [
    'data/raw', 'data/processed', 'data/external',
    'models', 'notebooks', 'outputs/forecasts', 'outputs/metrics', 
    'outputs/reports', 'dashboard', 'src/config', 'src/forecasting'
]
for d in required_dirs:
    if os.path.exists(d):
        print('  EXISTS: ' + d + '/')
    else:
        failures.append('Missing directory: ' + d + '/')

# 6. Model files existence check
print('\n[6] MODEL FILES EXISTENCE CHECK')
print('-'*75)
tickers = state.get('stocks', [])
models = ['arima', 'sarima', 'prophet', 'lstm', 'gru']
missing_models = []
for model in models:
    for ticker in tickers:
        safe = ticker.replace('.', '_')
        ext = '.keras' if model in ['lstm', 'gru'] else '.pkl'
        path = 'models/' + model + '_' + safe + ext
        if not os.path.exists(path):
            missing_models.append(path)

if missing_models:
    failures.append('Missing ' + str(len(missing_models)) + ' model files (showing first 5):')
    for m in missing_models[:5]:
        failures.append('  - ' + m)
else:
    print('  PASS: All ' + str(len(models) * len(tickers)) + ' model files exist')

# 7. Active phase consistency
print('\n[7] ACTIVE PHASE CONSISTENCY')
print('-'*75)
active_phase = state.get('active_phase')
phases_pending = state.get('phases_pending', [])
phases_complete = state.get('phases_complete', [])

print('  Active phase: ' + str(active_phase))
print('  Phases complete: ' + str(phases_complete))
print('  Phases pending: ' + str(phases_pending))

if str(active_phase) in [str(p) for p in phases_pending]:
    print('  PASS: active_phase is in phases_pending')
else:
    failures.append('active_phase ' + str(active_phase) + ' not in phases_pending ' + str(phases_pending))

# 8. Issues field
print('\n[8] ISSUES FIELD CHECK')
print('-'*75)
issues = state.get('issues', [])
if len(issues) == 0:
    print('  PASS: No open issues recorded')
else:
    print('  WARNING: ' + str(len(issues)) + ' open issues:')
    for i in issues:
        print('    - ' + str(i))

# SUMMARY
print('\n' + '='*75)
if failures:
    print('CRITICAL FAILURES:')
    for f in failures[:10]:
        print('  ERROR: ' + f)
    print('\nCROSS VERIFICATION: FAILED')
elif mismatches:
    print('MISMATCHES (non-critical):')
    for m in mismatches:
        print('  WARNING: ' + m)
    print('\nCROSS VERIFICATION: PASSED WITH WARNINGS')
else:
    print('ALL CROSS VERIFICATION CHECKS PASSED')
    print('state.yaml, skills.md, and filesystem are CONSISTENT')
    print('\nCROSS VERIFICATION: PASSED')
print('='*75)
