"""Quick script to run preprocessing phase."""
import sys
sys.path.insert(0, 'src')

from preprocessing.preprocessor import DataPreprocessor

p = DataPreprocessor()
print('Loading raw data...')
p.load_raw_data()

print(f"Loaded {len(p.train_data)} train datasets, {len(p.backtest_data)} backtest datasets")

print('Handling missing values...')
p.handle_missing_values()

print('Running ADF stationarity tests...')
results = p.test_stationarity()

print('=' * 60)
print('STATIONARITY TEST RESULTS (p < 0.05 = stationary)')
print('=' * 60)
for ticker, r in results.items():
    status = 'STATIONARY' if r['is_stationary'] else 'NON-STATIONARY'
    print(f'{ticker:20} p={r["p_value"]:.4f} [{status}]')

print()
print('Applying differencing to non-stationary series...')
p.apply_differencing()

print('Applying MinMaxScaler (fit on train only)...')
p.apply_scaling()

print('Saving processed datasets...')
p.save_processed_data()
print('Preprocessing complete.')
