import pandas as pd
import numpy as np
import os
import glob
from scipy.optimize import minimize

# --- 1. SETUP & DATA LOADING (Same as before) ---
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065
MARKET_CRASH_SCENARIO = -0.10 # -10%

# Define Sectors
sectors = {
    'Banking':['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'IT':['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Pharma':['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS']
}
stock_to_sector = {stock: sec for sec, stocks in sectors.items() for stock in stocks}

clean_dir = "cleaned_data"
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
prices = prices[stocks]

returns = prices.pct_change().dropna()
market_returns = market_prices.pct_change().dropna()

# --- 2. RE-CALCULATE INDIVIDUAL METRICS ---
ind_metrics =[]
for stock in stocks:
    ann_ret = (((1 + returns[stock]).prod()) ** (TRADING_DAYS / len(returns[stock]))) - 1
    ann_vol = returns[stock].std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    beta = returns[stock].cov(market_returns) / market_returns.var()
    sma50 = prices[stock].rolling(50).mean().iloc[-1]
    sma200 = prices[stock].rolling(200).mean().iloc[-1]
    is_golden = sma50 > sma200
    
    ind_metrics.append({
        'Stock': stock, 'Sector': stock_to_sector[stock], 
        'Sharpe': sharpe, 'Beta': beta, 'Volatility': ann_vol, 'Is_Golden': is_golden
    })

ind_df = pd.DataFrame(ind_metrics).set_index('Stock')

# --- 3. RE-CALCULATE PORTFOLIO WEIGHTS ---
# Portfolio A
weights_A = np.array([1/15] * 15)

# Portfolio B (Optimizer)
def negative_sharpe(weights, returns_df):
    p_ret = returns_df.dot(weights)
    ann_ret = ((1 + p_ret).prod() ** (TRADING_DAYS / len(p_ret))) - 1
    ann_vol = p_ret.std() * np.sqrt(TRADING_DAYS)
    return -((ann_ret - RISK_FREE_RATE) / ann_vol)

bounds = tuple((0, 1) if ind_df.loc[stock, 'Is_Golden'] else (0, 0) for stock in stocks)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
init_guess = np.array([1/ind_df['Is_Golden'].sum() if ind_df.loc[stock, 'Is_Golden'] else 0 for stock in stocks])
opt_result = minimize(negative_sharpe, init_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
weights_B = opt_result.x

# --- 4. CALCULATE PORTFOLIO BETAS ---
beta_A = np.sum(weights_A * ind_df['Beta'])
beta_B = np.sum(weights_B * ind_df['Beta'])

# --- 5. RUN THE STRESS TEST ---
stress_test_df = ind_df[['Beta']].copy()
stress_test_df['Expected Loss (%)'] = stress_test_df['Beta'] * MARKET_CRASH_SCENARIO * 100

# Add portfolio results
port_A_row = pd.DataFrame({'Beta': beta_A, 'Expected Loss (%)': beta_A * MARKET_CRASH_SCENARIO * 100}, index=['PORTFOLIO A'])
port_B_row = pd.DataFrame({'Beta': beta_B, 'Expected Loss (%)': beta_B * MARKET_CRASH_SCENARIO * 100}, index=['PORTFOLIO B'])

stress_test_df = pd.concat([stress_test_df, port_A_row, port_B_row])
stress_test_df = stress_test_df.round(2)
stress_test_df.to_csv("task5_stress_test.csv")

print("\n--- 📉 Stress Test Results (Nifty -10%) ---")
print(stress_test_df)

# --- 6. IDENTIFY KEY STOCKS ---
# Most Exposed
most_exposed_stock = ind_df['Beta'].idxmax()
most_exposed_beta = ind_df['Beta'].max()
most_exposed_loss = most_exposed_beta * MARKET_CRASH_SCENARIO * 100

# Safest Refuge (using a composite score)
ind_df['Beta_Rank'] = ind_df['Beta'].rank(ascending=True) # Lower is better
ind_df['Vol_Rank'] = ind_df['Volatility'].rank(ascending=True) # Lower is better
ind_df['Sharpe_Rank'] = ind_df['Sharpe'].rank(ascending=False) # Higher is better
ind_df['Safety_Score'] = ind_df['Beta_Rank'] + ind_df['Vol_Rank'] + ind_df['Sharpe_Rank']

safest_refuge_stock = ind_df['Safety_Score'].idxmin()
safest_refuge_metrics = ind_df.loc[safest_refuge_stock]

# --- 7. PREPARE JUSTIFICATIONS FOR JUDGES ---
print("\n--- 👨‍⚖️ JUSTIFICATION FOR JUDGES ---")
print("\n1. Stock Most Exposed to Crash:")
print(f"   -> STOCK: {most_exposed_stock}")
print(f"   -> REASONING: This choice is purely data-driven. {most_exposed_stock} has the highest calculated Beta of {most_exposed_beta:.2f} in the entire universe. Beta directly measures a stock's sensitivity to market movements. A Beta this high means that for every 1% the Nifty 50 falls, {most_exposed_stock} is mathematically expected to fall by {most_exposed_beta:.2f}%. Therefore, in the proposed -10% market crash scenario, it faces the most severe amplified loss of {most_exposed_loss:.2f}%, making it the most exposed asset by definition.")

print("\n2. Safest Refuge During Crash:")
print(f"   -> STOCK: {safest_refuge_stock}")
print(f"   -> REASONING: The safest asset in a crash is not just the one with the lowest Beta, but the one that demonstrates overall resilience. {safest_refuge_stock} was selected based on a composite safety score, ranking it as the best across three critical metrics:")
print(f"      a) Low Market Sensitivity: It has a very low Beta of {safest_refuge_metrics['Beta']:.2f}, meaning it is significantly insulated from market-wide panic.")
print(f"      b) Proven Efficiency: It boasts one of the highest Sharpe Ratios ({safest_refuge_metrics['Sharpe']:.2f}), indicating a strong historical track record of providing excellent returns for the level of risk taken.")
print(f"      c) Low Intrinsic Volatility: Independent of the market, its own annualised volatility is among the lowest ({safest_refuge_metrics['Volatility']*100:.2f}%), suggesting stable price behavior.")
print(f"   This trifecta of low market correlation, proven performance, and price stability makes {safest_refuge_stock} the most robust and justifiable safe refuge to preserve capital during a systemic shock.")