import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, "cleaned_data")
all_files = sorted(glob.glob(os.path.join(clean_dir, "*.csv")))
output_dir = os.path.join(base_dir, "outputs", "task3")
charts_dir = os.path.join(output_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)
analysis_end = pd.Timestamp("2024-12-31")

table_data =[]
crossover_events = {}

print("Computing SMAs and detecting crossovers...")

for file in all_files:
    ticker = os.path.basename(file).replace('.csv', '')
    if ticker == '^NSEI':
        continue
        
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    df = df.sort_index()
    df = df[df.index <= analysis_end].copy()
    if df.empty:
        continue

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Price'] = df[price_col]
    
    df['SMA50'] = df['Price'].rolling(window=50).mean()
    df['SMA200'] = df['Price'].rolling(window=200).mean()
    spread = df['SMA50'] - df['SMA200']
    prev_spread = spread.shift(1)

    valid_pair = spread.notna() & prev_spread.notna()
    golden_cross_mask = valid_pair & (prev_spread <= 0) & (spread > 0)
    death_cross_mask = valid_pair & (prev_spread >= 0) & (spread < 0)

    df['Crossover_Signal'] = 0
    df.loc[golden_cross_mask, 'Crossover_Signal'] = 2
    df.loc[death_cross_mask, 'Crossover_Signal'] = -2
    df['Trend'] = np.where(spread > 0, 1, np.where(spread < 0, -1, 0))
    df.loc[spread.isna(), 'Trend'] = np.nan
    
    crossover_events[ticker] = df.copy()
    
    last_day = df.iloc[-1]
    current_sma50 = last_day['SMA50']
    current_sma200 = last_day['SMA200']
    
    if pd.isna(current_sma50) or pd.isna(current_sma200):
        current_signal = "Neutral (Not enough data)"
    elif current_sma50 > current_sma200:
        current_signal = "Golden Cross"
    elif current_sma50 < current_sma200:
        current_signal = "Death Cross"
    else:
        current_signal = "Neutral"
        
    crossovers_only = df[df['Crossover_Signal'].isin([2.0, -2.0])]
    if not crossovers_only.empty:
        last_crossover_date = crossovers_only.index[-1].strftime('%Y-%m-%d')
    else:
        last_crossover_date = "No crossover in window"

    table_data.append({
        'Ticker': ticker,
        'SMA-50 Value': round(current_sma50, 2),
        'SMA-200 Value': round(current_sma200, 2),
        'Signal as of 31 Dec 2024': current_signal,
        'Date of Last Crossover': last_crossover_date
    })

signal_df = pd.DataFrame(table_data).sort_values('Ticker').reset_index(drop=True)
signal_table_path = os.path.join(output_dir, "task3_signal_table.csv")
signal_df.to_csv(signal_table_path, index=False)
print(f"\n[SUCCESS] Signal Table saved to '{signal_table_path}'\n")
print(signal_df.head())

stocks_to_plot = ["HDFCBANK.NS", "TCS.NS", "SUNPHARMA.NS"]

for ticker in stocks_to_plot:
    if ticker not in crossover_events:
        print(f"Skipping chart for {ticker}: data unavailable in analysis window")
        continue

    df_plot = crossover_events[ticker]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df_plot.index, df_plot['Price'], label='Price', color='black', alpha=0.5, linewidth=1)
    plt.plot(df_plot.index, df_plot['SMA50'], label='50-Day SMA', color='blue', linewidth=2)
    plt.plot(df_plot.index, df_plot['SMA200'], label='200-Day SMA', color='orange', linewidth=2)
    
    golden_crosses = df_plot[df_plot['Crossover_Signal'] == 2.0]
    plt.scatter(golden_crosses.index, golden_crosses['SMA50'], marker='^', color='green', s=150, label='Golden Cross', zorder=5)
    
    death_crosses = df_plot[df_plot['Crossover_Signal'] == -2.0]
    plt.scatter(death_crosses.index, death_crosses['SMA50'], marker='v', color='red', s=150, label='Death Cross', zorder=5)
    
    plt.title(f"{ticker} - SMA Crossover Analysis (2023-2024)")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    file_name = f"task3_chart_{ticker.replace('.NS', '')}.png"
    chart_path = os.path.join(charts_dir, file_name)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {chart_path}")

print(f"\nTask 3 visual outputs complete in '{output_dir}'")