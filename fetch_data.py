import yfinance as yf
import pandas as pd

tickers =[
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", # Banking
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",              # IT
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", # Pharma
    "^NSEI"                                                                 # Nifty 50
]

print("Downloading data...")
data = yf.download(tickers, start="2023-01-01", end="2024-12-31", auto_adjust=False)

# 3. Extract adjusted close prices (fallback to close if unavailable)
if 'Adj Close' in data.columns.get_level_values(0):
    prices_df = data['Adj Close']
else:
    prices_df = data['Close']

prices_df.to_csv("portfolio_adj_close.csv")
print("Data successfully downloaded and saved to 'portfolio_adj_close.csv'.")