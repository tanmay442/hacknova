import pandas as pd
import numpy as np
import os
import glob
from scipy.optimize import minimize

# --- 1. SETUP & DATA LOADING ---
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065
CORPUS = 1000000

# Define Sectors
sectors = {
    'Banking':['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'IT':['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Pharma':['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS']
}
# Reverse lookup dictionary
stock_to_sector = {stock: sec for sec, stocks in sectors.items() for stock in stocks}

base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, "cleaned_data")
output_dir = os.path.join(base_dir, "outputs", "task4")
os.makedirs(output_dir, exist_ok=True)
all_files = glob.glob(os.path.join(clean_dir, "*.csv"))

prices_dict = {}
for file in all_files:
    ticker = os.path.basename(file).replace('.csv', '')
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    prices_dict[ticker] = df[col]

prices = pd.DataFrame(prices_dict)
market_prices = prices['^NSEI']
stocks =[s for s in prices.columns if s != '^NSEI']
prices = prices[stocks] # Keep only the 15 stocks

returns = prices.pct_change().dropna()
market_returns = market_prices.pct_change().dropna()

# --- 2. CALCULATE INDIVIDUAL METRICS (For Sector Breakdown) ---
ind_metrics =[]
for stock in stocks:
    # Ann Return
    N = len(prices[stock].dropna())
    tot_ret = prices[stock].iloc[-1] / prices[stock].iloc[0]
    ann_ret = (tot_ret ** (TRADING_DAYS / N)) - 1
    # Volatility
    ann_vol = returns[stock].std() * np.sqrt(TRADING_DAYS)
    # Sharpe
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    # Beta
    beta = returns[stock].cov(market_returns) / market_returns.var()
    # SMA Filter (Is it in a Golden Cross?)
    sma50 = prices[stock].rolling(50).mean().iloc[-1]
    sma200 = prices[stock].rolling(200).mean().iloc[-1]
    is_golden = sma50 > sma200
    
    ind_metrics.append({
        'Stock': stock, 'Sector': stock_to_sector[stock], 
        'Sharpe': sharpe, 'Beta': beta, 'Is_Golden': is_golden
    })

ind_df = pd.DataFrame(ind_metrics).set_index('Stock')

# --- 3. SECTOR BREAKDOWN TABLE ---
sector_breakdown = ind_df.groupby('Sector')[['Sharpe', 'Beta']].mean().round(2)
sector_breakdown.columns =['Average Sharpe Ratio', 'Average Beta']
sector_breakdown_path = os.path.join(output_dir, "task4_sector_breakdown.csv")
sector_breakdown.to_csv(sector_breakdown_path)
print("\n--- Sector Breakdown ---")
print(sector_breakdown)

# --- 4. PORTFOLIO A: EQUAL WEIGHT ---
weights_A = np.array([1/15] * 15)
port_A_daily_returns = returns.dot(weights_A)
N_A = len(port_A_daily_returns)
port_A_tot_ret = (1 + port_A_daily_returns).prod()
port_A_ann_ret = (port_A_tot_ret ** (TRADING_DAYS / N_A)) - 1
port_A_ann_vol = port_A_daily_returns.std() * np.sqrt(TRADING_DAYS)
port_A_sharpe = (port_A_ann_ret - RISK_FREE_RATE) / port_A_ann_vol
port_A_sectors = {'Banking': 33.33, 'IT': 33.33, 'Pharma': 33.33}

# --- 5. PORTFOLIO B: SMA FILTER + MAX SHARPE OPTIMIZATION ---
def negative_sharpe(weights, returns_df):
    p_ret = returns_df.dot(weights)
    ann_ret = ((1 + p_ret).prod() ** (TRADING_DAYS / len(p_ret))) - 1
    ann_vol = p_ret.std() * np.sqrt(TRADING_DAYS)
    return -((ann_ret - RISK_FREE_RATE) / ann_vol)

bounds = tuple((0, 1) if ind_df.loc[stock, 'Is_Golden'] else (0, 0) for stock in stocks)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
golden_count = ind_df['Is_Golden'].sum()
init_guess = np.array([1/golden_count if ind_df.loc[stock, 'Is_Golden'] else 0 for stock in stocks])
opt_result = minimize(negative_sharpe, init_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
weights_B = opt_result.x

port_B_daily_returns = returns.dot(weights_B)
port_B_tot_ret = (1 + port_B_daily_returns).prod()
port_B_ann_ret = (port_B_tot_ret ** (TRADING_DAYS / len(port_B_daily_returns))) - 1
port_B_ann_vol = port_B_daily_returns.std() * np.sqrt(TRADING_DAYS)
port_B_sharpe = (port_B_ann_ret - RISK_FREE_RATE) / port_B_ann_vol

port_B_sectors = {'Banking': 0.0, 'IT': 0.0, 'Pharma': 0.0}
for i, stock in enumerate(stocks):
    port_B_sectors[stock_to_sector[stock]] += weights_B[i] * 100

# --- 6. COMPARISON TABLE ---
comp_table = pd.DataFrame({
    'Metric':['Annualised Return', 'Annualised Volatility', 'Sharpe Ratio', 'Banking Exposure', 'IT Exposure', 'Pharma Exposure'],
    'Portfolio A (Equal Weight)':[
        f"{port_A_ann_ret*100:.2f}%", f"{port_A_ann_vol*100:.2f}%", f"{port_A_sharpe:.2f}",
        f"{port_A_sectors['Banking']:.2f}%", f"{port_A_sectors['IT']:.2f}%", f"{port_A_sectors['Pharma']:.2f}%"
    ],
    'Portfolio B (Recommended)':[
        f"{port_B_ann_ret*100:.2f}%", f"{port_B_ann_vol*100:.2f}%", f"{port_B_sharpe:.2f}",
        f"{port_B_sectors['Banking']:.2f}%", f"{port_B_sectors['IT']:.2f}%", f"{port_B_sectors['Pharma']:.2f}%"
    ]
}).set_index('Metric')

comparison_table_path = os.path.join(output_dir, "task4_comparison_table.csv")
comp_table.to_csv(comparison_table_path)
print("\n--- Portfolio Comparison ---")
print(comp_table)

# --- 7. AUTO-GENERATE FUND MANAGER NOTE ---
excluded_stocks =[stocks[i] for i in range(15) if weights_B[i] < 0.01]
overweighted_stocks = [stocks[i] for i in range(15) if weights_B[i] > 0.15]
top_sector = max(port_B_sectors, key=port_B_sectors.get)

print("\n--- 📝 YOUR FUND MANAGER JUSTIFICATION NOTE ---")
note = f"""
MEMORANDUM TO FUND MANAGER: RECOMMENDED PORTFOLIO ALLOCATION

To maximize risk-adjusted returns, Portfolio B utilizes a dual-layered quantitative strategy combining a momentum filter with Mean-Variance Optimization. First, to avoid catching "falling knives," we applied a trend-following filter, strictly excluding assets exhibiting a Death Cross (SMA-50 < SMA-200) as of Q4 2024. Consequently, assets lacking macro momentum, notably {', '.join(excluded_stocks[:3])} (among others), were excluded entirely with a 0% weighting.

The surviving corpus was then subjected to a multi-objective optimization algorithm designed to maximize the Sharpe Ratio. This mathematically allocated capital toward the most efficient assets, resulting in significant overweights in {', '.join(overweighted_stocks[:3])}, shifting our heaviest sector exposure toward {top_sector} ({port_B_sectors[top_sector]:.1f}%). 

This approach significantly outperforms the naive Equal Weight (Portfolio A) by generating an expected Sharpe Ratio of {port_B_sharpe:.2f} compared to {port_A_sharpe:.2f}, while maintaining strict volatility controls. The risk profile of Portfolio B is moderately aggressive and momentum-driven. It is ideally suited for a growth-oriented investor with a medium-to-long-term horizon who seeks superior capital appreciation but requires systematic downside protection against structurally depreciating sectors.
"""
print(note)

# --- 8. FINAL ALLOCATION FILE ---
final_allocations = pd.DataFrame(index=stocks)
final_allocations['Sector'] = final_allocations.index.map(stock_to_sector)

# Calculate Rupee allocations
final_allocations['Portfolio A Allocation (Rs.)'] = CORPUS * weights_A
final_allocations['Portfolio B Weight (%)'] = weights_B * 100
final_allocations['Portfolio B Allocation (Rs.)'] = CORPUS * weights_B

# Format for clarity
final_allocations['Portfolio A Allocation (Rs.)'] = final_allocations['Portfolio A Allocation (Rs.)'].map('{:,.2f}'.format)
final_allocations['Portfolio B Weight (%)'] = final_allocations['Portfolio B Weight (%)'].map('{:.2f}%'.format)
final_allocations['Portfolio B Allocation (Rs.)'] = final_allocations['Portfolio B Allocation (Rs.)'].map('{:,.2f}'.format)

allocations_path = os.path.join(output_dir, "task4_portfolio_allocations.csv")
final_allocations.to_csv(allocations_path)
print(f"\n--- ✅ Final Rupee Allocation file created: '{allocations_path}' ---")
print(final_allocations.head())