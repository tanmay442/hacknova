import pandas as pd
import os
import glob

raw_dir = "raw_data"
clean_dir = "cleaned_data"
os.makedirs(clean_dir, exist_ok=True)

nifty_path = os.path.join(raw_dir, "^NSEI.csv")
nifty_df = pd.read_csv(nifty_path, index_col='Date', parse_dates=True)
market_dates = nifty_df.index

print("Cross-checking for >5 consecutive missing trading days...\n")

# Get all CSV files except Nifty 50
stock_files =[f for f in glob.glob(f"{raw_dir}/*.csv") if "^NSEI" not in f]

for file in stock_files:
    ticker = os.path.basename(file).replace('.csv', '')
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    
    # 2. Force the stock to match Nifty 50's dates. 
    # This exposes missing trading days as NaNs.
    df = df.reindex(market_dates)
    
    # 3. Calculate max consecutive missing days (using the 'Close' column as reference)
    is_na = df['Close'].isna()
    max_consecutive = is_na.groupby((~is_na).cumsum()).sum().max()
    
    if max_consecutive > 5:
        # Fails the requirement: Flag and do NOT save to cleaned_data
        print(f"[FLAGGED] {ticker}: {int(max_consecutive)} consecutive missing days. (Dropping asset)")
    else:
        # Passes the requirement: Handle <= 5 missing days
        if max_consecutive > 0:
            print(f"[FIXED]   {ticker}: {int(max_consecutive)} max missing days. (Forward filling)")
        else:
            print(f"[PERFECT] {ticker}: 0 missing days.")
            
        # Forward fill the acceptable gaps
        df_cleaned = df.ffill().bfill() 
        
        # Save the cleaned individual file
        df_cleaned.to_csv(os.path.join(clean_dir, f"{ticker}.csv"))

nifty_df.to_csv(os.path.join(clean_dir, "^NSEI.csv"))

print(f"\nCleaning complete! Ready-to-use files are in the '{clean_dir}' folder.")