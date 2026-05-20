"""Re-fetch all data with corrected column handling."""
import sys
sys.path.insert(0, 'src')
from data_fetching.fetcher import DataFetcher

f = DataFetcher()
print('Re-fetching all stocks with corrected column handling...')
f.fetch_all()
print('Fetch complete.')
print()
print('Verifying Close column types:')
for ticker in f.stocks:
    if ticker in f.data:
        df = f.data[ticker]
        close_type = df["Close"].dtype
        sample = df["Close"].iloc[0] if len(df) > 0 else "N/A"
        print(f"{ticker:20} {close_type} | sample: {sample}")
