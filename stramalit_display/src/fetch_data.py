import yfinance as yf
import pandas as pd
import os

tickers =[
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",              
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", 
    "^NSEI" 
]

output_dir = "raw_data"
os.makedirs(output_dir, exist_ok=True)

print("Downloading individual ticker data...")

for ticker in tickers:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start="2023-01-01", end="2025-01-01", auto_adjust=False)
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) # Drop the Ticker name row
            df.columns.name = None               # Remove the 'Price' label
        
        file_path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(file_path)
    else:
        print(f"  -> Warning: No data found for {ticker}")

print(f"\nAll done! Clean, flat individual files saved in '{output_dir}' folder.")