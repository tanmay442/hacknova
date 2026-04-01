import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Strategic Analytics Dashboard", layout="wide")

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065
CORPUS = 1000000
MARKET_CRASH_SCENARIO = -0.10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

sectors = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS']
}
stock_to_sector = {stock: sec for sec, stocks in sectors.items() for stock in stocks}

@st.cache_data
def load_data():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    price_dict = {}
    for file in all_files:
        ticker = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        col_to_use = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        price_dict[ticker] = df[col_to_use]
    return pd.DataFrame(price_dict)

prices = load_data()
common_prices = prices.dropna(how='any')
stocks = [c for c in common_prices.columns if c != '^NSEI']
returns = common_prices[stocks].pct_change().dropna(how='any')
market_return = common_prices['^NSEI'].pct_change().dropna(how='any')
market_var = market_return.var()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Risk Return Analysis", "Technical Signals", "Portfolio Optimization", "Chaos Stress Test"])

if page == "Risk Return Analysis":
    st.title("📊 Risk Return Analysis (Task 2)")
    
    metrics = []
    for stock in stocks:
        stock_prices = common_prices[stock]
        stock_returns = returns[stock]
        N = len(stock_returns)
        total_growth = stock_prices.iloc[-1] / stock_prices.iloc[0]
        ann_return = (total_growth ** (TRADING_DAYS / N)) - 1
        ann_vol = stock_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else np.nan
        cov = stock_returns.cov(market_return)
        beta = cov / market_var if market_var != 0 else np.nan
        drawdown = stock_prices / stock_prices.cummax() - 1
        max_dd = abs(drawdown.min())
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
    
    summary_df = pd.DataFrame(metrics).set_index('Stock')
    
    st.subheader("📋 Summary Table")
    st.dataframe(summary_df, width='stretch')
    
    st.subheader("📈 Risk-Return Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(summary_df['Annualised Volatility (%)'], summary_df['Annualised Return (%)'], color='blue', s=100)
    for i, txt in enumerate(summary_df.index):
        ax.annotate(txt, (summary_df['Annualised Volatility (%)'].iloc[i], summary_df['Annualised Return (%)'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Annualised Volatility (%) [RISK]')
    ax.set_ylabel('Annualised Return (%) [REWARD]')
    ax.set_title('Risk-Return Scatter Plot (2023-2024)')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.subheader("🔥 Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    corr_matrix = returns[stocks].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Heatmap of Daily Returns')
    st.pyplot(plt.gcf())
    plt.close()

elif page == "Technical Signals":
    st.title("📉 Technical Signals (Task 3)")
    
    analysis_end = pd.Timestamp("2024-12-31")
    table_data = []
    crossover_events = {}
    stocks_15 = [c for c in common_prices.columns if c != '^NSEI']
    
    BANKING_TICKERS = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
    IT_TICKERS = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
    PHARMA_TICKERS = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"]
    
    for ticker in stocks_15:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
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
            'Sector': stock_to_sector.get(ticker, 'N/A'),
            'SMA-50 Value': round(current_sma50, 2) if not pd.isna(current_sma50) else None,
            'SMA-200 Value': round(current_sma200, 2) if not pd.isna(current_sma200) else None,
            'Signal as of 31 Dec 2024': current_signal,
            'Date of Last Crossover': last_crossover_date
        })
    
    signal_df = pd.DataFrame(table_data).sort_values('Ticker').reset_index(drop=True)
    
    st.subheader("📋 Technical Signal Table")
    st.dataframe(signal_df, width='stretch')
    
    st.subheader("🎯 Sector Summary")
    sector_counts = signal_df.groupby('Sector')['Signal as of 31 Dec 2024'].value_counts().unstack(fill_value=0)
    st.dataframe(sector_counts)
    
    golden_cross_stocks = signal_df[signal_df['Signal as of 31 Dec 2024'] == 'Golden Cross']['Ticker'].tolist()
    death_cross_stocks = signal_df[signal_df['Signal as of 31 Dec 2024'] == 'Death Cross']['Ticker'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Golden Cross ({len(golden_cross_stocks)} stocks)**: {', '.join(golden_cross_stocks)}")
    with col2:
        st.error(f"**Death Cross ({len(death_cross_stocks)} stocks)**: {', '.join(death_cross_stocks)}")
    
    def pick_ticker(candidates, required_signal):
        matches = signal_df[
            signal_df['Ticker'].isin(candidates)
            & (signal_df['Signal as of 31 Dec 2024'] == required_signal)
            & (signal_df['Date of Last Crossover'] != 'No crossover in window')
        ].copy()
        if matches.empty:
            return None
        matches['CrossoverDate'] = pd.to_datetime(matches['Date of Last Crossover'], errors='coerce')
        matches = matches.dropna(subset=['CrossoverDate'])
        if matches.empty:
            return None
        return matches.sort_values('CrossoverDate', ascending=False).iloc[0]['Ticker']
    
    hdfc_row = signal_df[signal_df['Ticker'] == 'HDFCBANK.NS']
    if hdfc_row.empty or hdfc_row.iloc[0]['Date of Last Crossover'] == 'No crossover in window':
        st.warning("HDFCBANK.NS has no crossover in window - skipping charts")
    else:
        it_golden_ticker = pick_ticker(IT_TICKERS, 'Golden Cross')
        pharma_death_ticker = pick_ticker(PHARMA_TICKERS, 'Death Cross')
        
        stocks_to_plot = ["HDFCBANK.NS"]
        if it_golden_ticker:
            stocks_to_plot.append(it_golden_ticker)
        if pharma_death_ticker:
            stocks_to_plot.append(pharma_death_ticker)
        
        st.subheader("📊 SMA Crossover Charts")
        for ticker in stocks_to_plot:
            if ticker not in crossover_events:
                continue
            df_plot = crossover_events[ticker]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_plot.index, df_plot['Price'], label='Price', color='black', alpha=0.5, linewidth=1)
            ax.plot(df_plot.index, df_plot['SMA50'], label='50-Day SMA', color='blue', linewidth=2)
            ax.plot(df_plot.index, df_plot['SMA200'], label='200-Day SMA', color='orange', linewidth=2)
            
            golden_crosses = df_plot[df_plot['Crossover_Signal'] == 2.0]
            ax.scatter(golden_crosses.index, golden_crosses['SMA50'], marker='^', color='green', s=150, label='Golden Cross', zorder=5)
            
            death_crosses = df_plot[df_plot['Crossover_Signal'] == -2.0]
            ax.scatter(death_crosses.index, death_crosses['SMA50'], marker='v', color='red', s=150, label='Death Cross', zorder=5)
            
            ax.set_title(f"{ticker} - SMA Crossover Analysis (2023-2024)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

elif page == "Portfolio Optimization":
    st.title("💼 Portfolio Optimization (Task 4)")
    
    stocks_15 = [s for s in common_prices.columns if s != '^NSEI']
    returns_15 = common_prices[stocks_15].pct_change().dropna(how='any')
    market_returns = common_prices['^NSEI'].pct_change().dropna(how='any')
    
    ind_metrics = []
    for stock in stocks_15:
        N = len(common_prices[stock].dropna())
        tot_ret = common_prices[stock].iloc[-1] / common_prices[stock].iloc[0]
        ann_ret = (tot_ret ** (TRADING_DAYS / N)) - 1
        ann_vol = returns_15[stock].std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
        beta = returns_15[stock].cov(market_returns) / market_returns.var()
        sma50 = common_prices[stock].rolling(50).mean().iloc[-1]
        sma200 = common_prices[stock].rolling(200).mean().iloc[-1]
        is_golden = sma50 > sma200
        
        ind_metrics.append({
            'Stock': stock,
            'Sector': stock_to_sector[stock],
            'Annualised Return (%)': round(ann_ret * 100, 2),
            'Annualised Volatility (%)': round(ann_vol * 100, 2),
            'Sharpe': round(sharpe, 2),
            'Beta': round(beta, 2),
            'Is_Golden': is_golden
        })
    
    ind_df = pd.DataFrame(ind_metrics).set_index('Stock')
    
    weights_A = np.array([1/15] * 15)
    port_A_daily_returns = returns_15.dot(weights_A)
    N_A = len(port_A_daily_returns)
    port_A_tot_ret = (1 + port_A_daily_returns).prod()
    port_A_ann_ret = (port_A_tot_ret ** (TRADING_DAYS / N_A)) - 1
    port_A_ann_vol = port_A_daily_returns.std() * np.sqrt(TRADING_DAYS)
    port_A_sharpe = (port_A_ann_ret - RISK_FREE_RATE) / port_A_ann_vol
    
    def negative_sharpe(weights, returns_df):
        p_ret = returns_df.dot(weights)
        ann_ret = ((1 + p_ret).prod() ** (TRADING_DAYS / len(p_ret))) - 1
        ann_vol = p_ret.std() * np.sqrt(TRADING_DAYS)
        return -((ann_ret - RISK_FREE_RATE) / ann_vol)
    
    bounds = tuple((0, 1) if ind_df.loc[stock, 'Is_Golden'] else (0, 0) for stock in stocks_15)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    golden_count = ind_df['Is_Golden'].sum()
    init_guess = np.array([1/golden_count if ind_df.loc[stock, 'Is_Golden'] else 0 for stock in stocks_15])
    opt_result = minimize(negative_sharpe, init_guess, args=(returns_15,), method='SLSQP', bounds=bounds, constraints=constraints)
    weights_B = opt_result.x
    
    port_B_daily_returns = returns_15.dot(weights_B)
    port_B_ann_ret = ((1 + port_B_daily_returns).prod() ** (TRADING_DAYS / len(port_B_daily_returns))) - 1
    port_B_ann_vol = port_B_daily_returns.std() * np.sqrt(TRADING_DAYS)
    port_B_sharpe = (port_B_ann_ret - RISK_FREE_RATE) / port_B_ann_vol
    
    port_B_sectors = {'Banking': 0.0, 'IT': 0.0, 'Pharma': 0.0}
    for i, stock in enumerate(stocks_15):
        port_B_sectors[stock_to_sector[stock]] += weights_B[i] * 100
    
    st.subheader("📊 Portfolio Comparison")
    comp_data = {
        'Metric': ['Annualised Return', 'Annualised Volatility', 'Sharpe Ratio'],
        'Portfolio A (Equal Weight)': [f"{port_A_ann_ret*100:.2f}%", f"{port_A_ann_vol*100:.2f}%", f"{port_A_sharpe:.2f}"],
        'Portfolio B (Optimized)': [f"{port_B_ann_ret*100:.2f}%", f"{port_B_ann_vol*100:.2f}%", f"{port_B_sharpe:.2f}"]
    }
    comp_df = pd.DataFrame(comp_data).set_index('Metric')
    st.dataframe(comp_df, width='stretch')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Performance Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_plot = ['Return (%)', 'Volatility (%)', 'Sharpe']
        val_a = [port_A_ann_ret * 100, port_A_ann_vol * 100, port_A_sharpe]
        val_b = [port_B_ann_ret * 100, port_B_ann_vol * 100, port_B_sharpe]
        x = np.arange(len(metrics_plot))
        width_bar = 0.35
        ax.bar(x - width_bar/2, val_a, width_bar, label='Portfolio A', color='#9ca3af')
        ax.bar(x + width_bar/2, val_b, width_bar, label='Portfolio B', color='#0ea5e9')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_plot)
        ax.legend()
        ax.set_title('Portfolio Performance Comparison')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🥧 Portfolio B Sector Allocation")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(port_B_sectors.values(), labels=port_B_sectors.keys(), autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_title('Portfolio B Sector Exposure')
        st.pyplot(fig2)
    
    st.subheader("📋 Individual Stock Metrics")
    st.dataframe(ind_df[['Sector', 'Annualised Return (%)', 'Annualised Volatility (%)', 'Sharpe', 'Beta', 'Is_Golden']], width='stretch')
    
    st.subheader("💰 Recommended Portfolio Allocations (₹10 Lakhs)")
    alloc_df = pd.DataFrame({
        'Stock': stocks_15,
        'Sector': [stock_to_sector[s] for s in stocks_15],
        'Weight (%)': [f"{w*100:.2f}%" for w in weights_B],
        'Allocation (₹)': [f"₹{CORPUS * w:,.0f}" for w in weights_B]
    }).set_index('Stock')
    st.dataframe(alloc_df, width='stretch')

elif page == "Chaos Stress Test":
    st.title("⚡ Chaos Stress Test (Task 5)")
    
    ind_metrics = []
    market_returns = common_prices['^NSEI'].pct_change().dropna(how='any')
    stocks_15 = [s for s in common_prices.columns if s != '^NSEI']
    returns_15 = common_prices[stocks_15].pct_change().dropna(how='any')
    
    for stock in stocks_15:
        ann_ret = (((1 + returns_15[stock]).prod()) ** (TRADING_DAYS / len(returns_15[stock]))) - 1
        ann_vol = returns_15[stock].std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
        beta = returns_15[stock].cov(market_returns) / market_returns.var()
        sma50 = common_prices[stock].rolling(50).mean().iloc[-1]
        sma200 = common_prices[stock].rolling(200).mean().iloc[-1]
        is_golden = sma50 > sma200
        
        ind_metrics.append({
            'Stock': stock, 'Sector': stock_to_sector[stock],
            'Sharpe': sharpe, 'Beta': beta, 'Volatility': ann_vol, 'Is_Golden': is_golden
        })
    
    ind_df = pd.DataFrame(ind_metrics).set_index('Stock')
    
    weights_A = np.array([1/15] * 15)
    
    def negative_sharpe(weights, returns_df):
        p_ret = returns_df.dot(weights)
        ann_ret = ((1 + p_ret).prod() ** (TRADING_DAYS / len(p_ret))) - 1
        ann_vol = p_ret.std() * np.sqrt(TRADING_DAYS)
        return -((ann_ret - RISK_FREE_RATE) / ann_vol)
    
    bounds = tuple((0, 1) if ind_df.loc[stock, 'Is_Golden'] else (0, 0) for stock in stocks_15)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    golden_count = ind_df['Is_Golden'].sum()
    init_guess = np.array([1/golden_count if ind_df.loc[stock, 'Is_Golden'] else 0 for stock in stocks_15])
    opt_result = minimize(negative_sharpe, init_guess, args=(returns_15,), method='SLSQP', bounds=bounds, constraints=constraints)
    weights_B = opt_result.x
    
    beta_A = np.sum(weights_A * ind_df['Beta'])
    beta_B = np.sum(weights_B * ind_df['Beta'])
    
    stress_test_df = ind_df[['Beta']].copy()
    stress_test_df['Expected Loss (%)'] = (stress_test_df['Beta'] * MARKET_CRASH_SCENARIO * 100).round(2)
    stress_test_df = stress_test_df.round(2)
    
    st.subheader("📉 Stress Test Results (Nifty -10% Scenario)")
    st.dataframe(stress_test_df, width='stretch')
    
    most_exposed_stock = ind_df['Beta'].idxmax()
    most_exposed_beta = ind_df['Beta'].max()
    most_exposed_loss = most_exposed_beta * MARKET_CRASH_SCENARIO * 100
    
    ind_df['Beta_Rank'] = ind_df['Beta'].rank(ascending=True)
    ind_df['Vol_Rank'] = ind_df['Volatility'].rank(ascending=True)
    ind_df['Sharpe_Rank'] = ind_df['Sharpe'].rank(ascending=False)
    ind_df['Safety_Score'] = ind_df['Beta_Rank'] + ind_df['Vol_Rank'] + ind_df['Sharpe_Rank']
    safest_refuge_stock = ind_df['Safety_Score'].idxmin()
    safest_refuge_metrics = ind_df.loc[safest_refuge_stock]
    
    col1, col2 = st.columns(2)
    with col1:
        st.error(f"### 🚨 Most Exposed: {most_exposed_stock}")
        st.write(f"**Beta:** {most_exposed_beta:.2f}")
        st.write(f"**Expected Loss:** {most_exposed_loss:.2f}%")
        st.write("*Highest market sensitivity - will fall most in a crash*")
    
    with col2:
        st.success(f"### 🛡️ Safest Refuge: {safest_refuge_stock}")
        st.write(f"**Beta:** {safest_refuge_metrics['Beta']:.2f}")
        st.write(f"**Sharpe:** {safest_refuge_metrics['Sharpe']:.2f}")
        st.write(f"**Volatility:** {safest_refuge_metrics['Volatility']*100:.2f}%")
        st.write("*Best composite safety score - best capital protector*")
    
    st.subheader("📊 Portfolio Beta Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    port_names = ['Portfolio A', 'Portfolio B']
    betas = [beta_A, beta_B]
    colors = ['#9ca3af', '#0ea5e9']
    ax.bar(port_names, betas, color=colors)
    ax.set_ylabel('Portfolio Beta')
    ax.set_title('Portfolio Beta - Lower is Safer')
    for i, v in enumerate(betas):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    st.pyplot(fig)
    
    st.info("### 📝 Key Insight")
    st.write(f"Portfolio B has a **lower beta ({beta_B:.2f})** than Portfolio A ({beta_A:.2f}), making it **less vulnerable** to market crashes. The optimized portfolio not only delivers higher returns but also provides better downside protection.")