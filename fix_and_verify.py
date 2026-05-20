import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import joblib
import os
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

preprocessor = DataPreprocessor()
preprocessor.load_raw_data()

tickers = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','ITC.NS',
           'MARUTI.NS','SUNPHARMA.NS','TATASTEEL.NS','BAJFINANCE.NS']

print('Updating ARIMA metrics for RELIANCE and INFY...')

# Load existing ARIMA metrics
arima_metrics = pd.read_csv('outputs/metrics/arima_metrics.csv')

# Update RELIANCE and INFY
for ticker in ['RELIANCE.NS', 'INFY.NS']:
    safe = ticker.replace('.', '_')
    model = joblib.load('models/arima_' + safe + '.pkl')
    backtest = preprocessor.backtest_data[ticker]['Close'].squeeze()
    
    fc = model.predict(n_periods=len(backtest))
    
    evaluator = ModelEvaluator()
    m = evaluator.compute_single_metrics(backtest.values, fc)
    
    # Update in dataframe
    idx = arima_metrics[arima_metrics['ticker'] == ticker].index
    if len(idx) > 0:
        arima_metrics.loc[idx, 'rmse'] = m['rmse']
        arima_metrics.loc[idx, 'mae'] = m['mae']
        arima_metrics.loc[idx, 'mape'] = m['mape']
        arima_metrics.loc[idx, 'directional_accuracy'] = m['directional_accuracy']
        print(ticker + ': DA=' + str(round(m['directional_accuracy'],1)) + '%')

arima_metrics.to_csv('outputs/metrics/arima_metrics.csv', index=False)
print('ARIMA metrics updated')

# Regenerate model_comparison.csv
print('\nRegenerating model_comparison.csv...')
files = ['arima', 'sarima', 'holt_winters', 'prophet', 'lstm', 'gru']
all_data = []
for f in files:
    try:
        df = pd.read_csv('outputs/metrics/' + f + '_metrics.csv')
        all_data.append(df)
    except:
        pass

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv('outputs/metrics/model_comparison.csv', index=False)
    print('model_comparison.csv created: ' + str(len(combined)) + ' rows')
    
    # Also create model_comparison_metrics.csv for compatibility
    combined.to_csv('outputs/metrics/model_comparison_metrics.csv', index=False)
    print('model_comparison_metrics.csv created')
    
    zero_da = combined[combined['directional_accuracy'] == 0]
    print('Models with 0% DA: ' + str(len(zero_da)))

print('\nDone. Running final verification...')
