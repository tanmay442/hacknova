import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Constants from the assignment
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065

print("Loading cleaned data...")
clean_dir = "cleaned_data"
all_files = sorted(glob.glob(os.path.join(clean_dir, "*.csv")))

# 1. Load all data into a single DataFrame of Adjusted Close prices
price_dict = {}
for file in all_files:
    ticker = os.path.basename(file).replace('.csv', '')
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    # Use Adj Close for accurate return calculations as requested
    col_to_use = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    price_dict[ticker] = df[col_to_use]

prices = pd.DataFrame(price_dict)
if '^NSEI' not in prices.columns:
    raise ValueError("^NSEI market benchmark file is missing in cleaned_data.")

# Use the full common window across all assets for consistent metric comparisons.
common_prices = prices.dropna(how='any')
returns = common_prices.pct_change().dropna(how='any')
print(
    f"Analysis window: {common_prices.index.min().date()} to "
    f"{common_prices.index.max().date()} ({len(common_prices)} trading days)"
)

market_return = returns['^NSEI']
market_var = market_return.var()

# Get only the 15 stocks (exclude Nifty 50 from the final table)
stocks = [c for c in common_prices.columns if c != '^NSEI']

metrics =[]

print("Calculating metrics...")
for stock in stocks:
    stock_prices = common_prices[stock]
    stock_returns = returns[stock]
    
    # 1. Annualised Return (CAGR approach scaled to 1 year)
    N = len(stock_returns)
    total_growth = stock_prices.iloc[-1] / stock_prices.iloc[0]
    ann_return = (total_growth ** (TRADING_DAYS / N)) - 1
    
    # 2. Annualised Volatility
    ann_vol = stock_returns.std() * np.sqrt(TRADING_DAYS)
    
    # 3. Sharpe Ratio
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else np.nan
    
    # 4. Beta vs Nifty 50
    cov = stock_returns.cov(market_return)
    beta = cov / market_var if market_var != 0 else np.nan
    
    # 5. Maximum Drawdown
    drawdown = stock_prices / stock_prices.cummax() - 1
    max_dd = abs(drawdown.min())
    
    # 6. 50-Day & 200-Day SMA Trend Signal (as of last day)
    sma50 = stock_prices.rolling(window=50).mean().iloc[-1]
    sma200 = stock_prices.rolling(window=200).mean().iloc[-1]
    trend_signal = "Golden Cross" if sma50 > sma200 else "Death Cross"
    
    metrics.append({
        'Stock': stock,
        'Annualised Return (%)': round(ann_return * 100, 2),
        'Annualised Volatility (%)': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Beta vs Nifty 50': round(beta, 2),
        'Maximum Drawdown (%)': round(max_dd * 100, 2),
        '50/200 SMA Signal': trend_signal
    })

# --- TASK 2 OUTPUTS ---

# Output 1: Summary Table
summary_df = pd.DataFrame(metrics).set_index('Stock')
summary_df.to_csv("task2_summary_table.csv")
print("\n--- Summary Table saved to 'task2_summary_table.csv' ---")
print(summary_df.head()) # Preview

# Output 2: Risk-Return Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(summary_df['Annualised Volatility (%)'], summary_df['Annualised Return (%)'], color='blue')

# Label each stock on the plot
for i, txt in enumerate(summary_df.index):
    plt.annotate(txt, 
                 (summary_df['Annualised Volatility (%)'].iloc[i], summary_df['Annualised Return (%)'].iloc[i]),
                 xytext=(5,5), textcoords='offset points', fontsize=8)

plt.title('Risk-Return Scatter Plot (2023-2024)')
plt.xlabel('Annualised Volatility (%) [RISK]')
plt.ylabel('Annualised Return (%) [REWARD]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("task2_scatter_plot.png", dpi=300, bbox_inches='tight')
print("--- Scatter plot saved to 'task2_scatter_plot.png' ---")

# Output 3: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = returns[stocks].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of Daily Returns across 15 Stocks')
plt.savefig("task2_correlation_heatmap.png", dpi=300, bbox_inches='tight')
print("--- Heatmap saved to 'task2_correlation_heatmap.png' ---")

# Helper block for the written assignment: Find most correlated pair
corr_values = corr_matrix.to_numpy(copy=True)
np.fill_diagonal(corr_values, np.nan)
max_idx = np.unravel_index(np.nanargmax(np.abs(corr_values)), corr_values.shape)
row_idx = int(max_idx[0])
col_idx = int(max_idx[1])
stock_a = corr_matrix.index[row_idx]
stock_b = corr_matrix.columns[col_idx]
top_corr = float(corr_values[row_idx, col_idx])
print(
    "\n[HELPER FOR WRITTEN TASK] The most highly correlated pair is: "
    f"{(stock_a, stock_b)} with a correlation of {top_corr:.3f}"
)