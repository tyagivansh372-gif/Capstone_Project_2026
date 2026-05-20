import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import joblib
from preprocessing.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator

preprocessor = DataPreprocessor()
preprocessor.load_raw_data()

tickers = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','ITC.NS',
           'MARUTI.NS','SUNPHARMA.NS','TATASTEEL.NS','BAJFINANCE.NS']

metrics_list = []

for ticker in tickers:
    safe = ticker.replace('.', '_')
    try:
        model = joblib.load(f'models/arima_{safe}.pkl')
        backtest = preprocessor.backtest_data[ticker]['Close'].squeeze()
        
        steps = len(backtest)
        fc_series = model.predict(n_periods=steps)
        fc = fc_series.values if hasattr(fc_series, 'values') else fc_series
        
        evaluator = ModelEvaluator()
        m = evaluator.compute_single_metrics(backtest.values, fc)
        m['ticker'] = ticker
        m['model'] = 'ARIMA'
        metrics_list.append(m)
        
        mape = m['mape']
        da = m['directional_accuracy']
        print(f'{ticker}: MAPE={mape:.2f}%, DA={da:.1f}%')
    except Exception as e:
        print(f'{ticker}: ERROR - {e}')

# Save updated metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('outputs/metrics/arima_metrics.csv', index=False)
print('\nUpdated ARIMA metrics saved')
