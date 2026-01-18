import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import timedelta, datetime
from scipy.stats import norm
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroQuant Pro Terminal", layout="wide")
st.title("✈️ Aerospace Quantitative Terminal")

# Initialize VADER
nltk.download('vader_lexicon', quiet=True)

# ==========================================
# 1. ROBUST DATA FETCHING
# ==========================================

@st.cache_data(ttl=60)
def get_stock_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "Free Cash Flow": info.get("freeCashflow", None),
            "Shares Outstanding": info.get("sharesOutstanding", None)
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_market_overview():
    market_tickers = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225', 'GBPUSD=X', 'EURUSD=X', 'JPY=X', 'GC=F', 'CL=F']
    try:
        market_data = yf.download(market_tickers, period="5d", progress=False)['Close']
        market_data = market_data.ffill().dropna()
        
        if len(market_data) >= 2:
            latest = market_data.iloc[-1]
            prev = market_data.iloc[-2]
        else:
            latest = market_data.iloc[-1]
            prev = latest
        
        change_pct = ((latest - prev) / prev) * 100
        
        stock_universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AMD', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'AIR.PA', 'RR.L', 'JPM', 'BAC', 'GS', 'HSBC', 'XOM', 'CVX', 'SHEL', 'PFE', 'LLY', 'JNJ', 'F', 'GM', 'TM', 'PLTR', 'COIN', 'HOOD', 'DIS', 'NFLX']
        stocks_data = yf.download(stock_universe, period="5d", progress=False)['Close']
        stocks_data = stocks_data.ffill().dropna()
        
        if len(stocks_data) >= 2:
            stock_latest = stocks_data.iloc[-1]
            stock_prev = stocks_data.iloc[-2]
            stock_change = ((stock_latest - stock_prev) / stock_prev) * 100
        else:
            stock_latest = stocks_data.iloc[-1]
            stock_change = pd.Series(0, index=stock_latest.index)
            
        movers_df = pd.DataFrame({'Price': stock_latest, 'Change (%)': stock_change})
        return latest, change_pct, movers_df
    except Exception as e:
        return pd.Series(), pd.Series(), pd.DataFrame()

@st.cache_data(ttl=3600)
def get_insider_trading():
    try:
        url = 'https://finviz.com/insidertrading.ashx'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        html = BeautifulSoup(response.text, features='html.parser')
        target_table = None
        for t in html.findAll('table'):
            if len(t.findAll('tr')) > 10:
                target_table = t
                break
        rows = []
        if target_table:
            for row in target_table.findAll('tr')[1:11]: 
                cols = row.findAll('td')
                if len(cols) > 4:
                    rows.append([cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip(), cols[4].text.strip(), cols[5].text.strip(), cols[6].text.strip()])
            return pd.DataFrame(rows, columns=['Ticker', 'Owner', 'Relation', 'Date', 'Transaction', 'Cost', 'Shares'])
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_buffett_portfolio():
    try:
        url = 'https://www.dataroma.com/m/holdings.php?m=BRK'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        html = BeautifulSoup(response.text, features='html.parser')
        table = html.find('table', id='grid')
        rows = []
        if table:
            for row in table.findAll('tr')[1:]:
                cols = row.findAll('td')
                if len(cols) > 1:
                    rows.append([cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip()])
        return pd.DataFrame(rows, columns=['Ticker', 'Company', '% of Portfolio'])
    except Exception:
        return pd.DataFrame()

# ==========================================
# 2. ANALYSIS ALGORITHMS
# ==========================================

def identify_candlestick_patterns(data):
    df = data.copy()
    for col in ['Pattern_Bullish_Engulfing', 'Pattern_Bearish_Engulfing', 'Pattern_Hammer', 'Pattern_Doji']:
        df[col] = False
    df['Body'] = df['Close'] - df['Open']
    df['Body_Size'] = df['Body'].abs()
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Range'] = df['High'] - df['Low']
    df['Pattern_Doji'] = df['Body_Size'] <= (0.1 * df['Range'])
    df['Pattern_Hammer'] = (df['Lower_Wick'] >= 2 * df['Body_Size']) & (df['Upper_Wick'] <= 0.5 * df['Body_Size']) & (df['Body_Size'] >= 0.05 * df['Range'])
    df['Prev_Body'] = df['Body'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Pattern_Bullish_Engulfing'] = (df['Prev_Body'] < 0) & (df['Body'] > 0) & (df['Close'] > df['Prev_Open']) & (df['Open'] < df['Prev_Close'])
    df['Pattern_Bearish_Engulfing'] = (df['Prev_Body'] > 0) & (df['Body'] < 0) & (df['Close'] < df['Prev_Open']) & (df['Open'] > df['Prev_Close'])
    return df

def identify_macro_patterns(data, window=5):
    df = data.copy()
    for col in ['Pattern_HeadShoulders', 'Pattern_InvHeadShoulders', 'Pattern_DoubleTop', 'Pattern_Wedge']:
        df[col] = False
    df['Max'] = df['High'].rolling(window=window*2+1, center=True).max()
    df['Min'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['is_Pivot_High'] = (df['High'] == df['Max'])
    df['is_Pivot_Low'] = (df['Low'] == df['Min'])
    
    last_highs = df[df['is_Pivot_High']].tail(3)
    if len(last_highs) == 3:
        p1, p2, p3 = last_highs['High'].values
        if (p2 > p1) and (p2 > p3) and (abs(p1 - p3) / p1 < 0.02):
            df.loc[last_highs.index[-1], 'Pattern_HeadShoulders'] = True

    last_lows = df[df['is_Pivot_Low']].tail(3)
    if len(last_lows) == 3:
        p1, p2, p3 = last_lows['Low'].values
        if (p2 < p1) and (p2 < p3) and (abs(p1 - p3) / p1 < 0.02):
            df.loc[last_lows.index[-1], 'Pattern_InvHeadShoulders'] = True
            
    last_highs_2 = df[df['is_Pivot_High']].tail(2)
    if len(last_highs_2) == 2:
        p1, p2 = last_highs_2['High'].values
        if abs(p1 - p2) / p1 < 0.01:
            df.loc[last_highs_2.index[-1], 'Pattern_DoubleTop'] = True

    last_2_highs = df[df['is_Pivot_High']].tail(2)['High'].values
    last_2_lows = df[df['is_Pivot_Low']].tail(2)['Low'].values
    if len(last_2_highs) == 2 and len(last_2_lows) == 2:
        h_slope_down = last_2_highs[1] < last_2_highs[0]
        l_slope_up = last_2_lows[1] > last_2_lows[0]
        if h_slope_down and l_slope_up:
            df.loc[df.index[-1], 'Pattern_Wedge'] = True
    return df

def find_support_resistance(data, window=10):
    df = data.copy()
    df['Min'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['Max'] = df['High'].rolling(window=window*2+1, center=True).max()
    pivots = df[df['Low'] == df['Min']]['Low'].tolist() + df[df['High'] == df['Max']]['High'].tolist()
    pivots.sort()
    if not pivots: return []
    clusters = []
    current_cluster = [pivots[0]]
    for p in pivots[1:]:
        if p <= current_cluster[0] * 1.015: current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    return [np.mean(c) for c in clusters if len(c) >= 3]

def calculate_indicators(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['StdDev'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['StdDev'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['StdDev'])
    v = data['Volume'].values
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (tp * v).cumsum() / v.cumsum()
    high9 = data['High'].rolling(window=9).max(); low9 = data['Low'].rolling(window=9).min()
    data['Tenkan'] = (high9 + low9) / 2
    high26 = data['High'].rolling(window=26).max(); low26 = data['Low'].rolling(window=26).min()
    data['Kijun'] = (high26 + low26) / 2
    data['SpanA'] = ((data['Tenkan'] + data['Kijun']) / 2).shift(26)
    high52 = data['High'].rolling(window=52).max(); low52 = data['Low'].rolling(window=52).min()
    data['SpanB'] = ((high52 + low52) / 2).shift(26)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD_Line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    low14 = data['Low'].rolling(window=14).min(); high14 = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * ((data['Close'] - low14) / (high14 - low14))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return data

def calculate_quant_metrics(data):
    if len(data) < 2: return 0, 0
    returns = data['Close'].pct_change().dropna()
    mean_return = returns.mean() * 252
    volatility = returns.std() * (252**0.5)
    sharpe = mean_return / volatility if volatility != 0 else 0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() * 100
    return sharpe, max_drawdown

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate Bollinger Bands specifically for the Scanner
def calculate_bollinger_bands_val(data, window=20):
    mid = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = mid + (2 * std)
    lower = mid - (2 * std)
    return upper, lower

def monte_carlo_simulation(ticker, days_ahead, simulations=200, vol_scale=1.0, backtest=False, drift_method="Historical"):
    try:
        sim_data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if sim_data.empty: return None, None, None, None, None, None
        if isinstance(sim_data.columns, pd.MultiIndex):
            sim_data.columns = sim_data.columns.get_level_values(0)
        
        if backtest:
            if len(sim_data) <= days_ahead: return None, None, None, None, None, None
            actual_path = sim_data['Close'].iloc[-days_ahead:]
            sim_data = sim_data.iloc[:-days_ahead]
        else:
            actual_path = None

        log_returns = np.log(1 + sim_data['Close'].pct_change()).dropna()
        if log_returns.empty: return None, None, None, None, None, None

        sigma = log_returns.std() * vol_scale
        start_price = sim_data['Close'].iloc[-1]
        
        log_prices = np.log(sim_data['Close'])
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)
        future_x = np.arange(len(log_prices), len(log_prices) + days_ahead + 1)
        linear_line = np.exp(intercept + slope * future_x)

        if drift_method == "Linear Regression Trend": mu = slope 
        else: mu = log_returns.mean()

        drift = mu - (0.5 * sigma**2)
        dt = 1
        shock = np.random.normal(0, 1, (days_ahead, simulations))
        daily_returns = np.exp(drift * dt + sigma * np.sqrt(dt) * shock)
        
        price_paths = np.zeros((days_ahead + 1, simulations))
        price_paths[0] = start_price
        for t in range(1, days_ahead + 1):
            price_paths[t] = price_paths[t-1] * daily_returns[t-1]
            
        last_date = sim_data.index[-1]
        if hasattr(last_date, 'date'): last_date = last_date.date()
        future_dates = [last_date + timedelta(days=i) for i in range(days_ahead + 1)]
        
        lower_bound_path = np.percentile(price_paths, 5, axis=1)
        upper_bound_path = np.percentile(price_paths, 95, axis=1)
        
        return future_dates, price_paths, actual_path, linear_line, lower_bound_path, upper_bound_path
    except Exception:
        return None, None, None, None, None, None

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Market Overview", 
    "📰 News & Intelligence", 
    "📈 Pro Charting", 
    "🔍 Sector Research",
    "🐋 Competitor Research",
    "📊 Portfolio Analysis",
    "💰 Valuation & Risk"
])

# TAB 1: MARKET OVERVIEW
with tab1:
    st.header("Global Market Monitor")
    try:
        prices, changes, movers = get_market_overview()
        st.subheader("Major Indices & Commodities")
        c1, c2, c3, c4, c5 = st.columns(5)
        def display_metric(col, ticker, name):
            if ticker in prices:
                col.metric(name, f"{prices[ticker]:,.2f}", f"{changes[ticker]:+.2f}%")
            else:
                col.metric(name, "N/A", "N/A")
        indices_map = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ', '^FTSE': 'FTSE 100', 'GC=F': 'Gold'}
        cols = [c1, c2, c3, c4, c5]
        for i, (ticker, name) in enumerate(indices_map.items()):
            display_metric(cols[i], ticker, name)
        st.divider()
        st.subheader("Forex (Currencies)")
        c1, c2, c3, c4, c5 = st.columns(5)
        forex_map = {'GBPUSD=X': 'GBP / USD', 'EURUSD=X': 'EUR / USD', 'JPY=X': 'USD / JPY', 'CL=F': 'Crude Oil', '^N225': 'Nikkei 225'}
        cols = [c1, c2, c3, c4, c5]
        for i, (ticker, name) in enumerate(forex_map.items()):
            display_metric(cols[i], ticker, name)
        st.divider()
        st.subheader("Top Daily Movers (Major Stocks)")
        if not movers.empty:
            col_gain, col_loss = st.columns(2)
            gainers = movers.sort_values(by='Change (%)', ascending=False).head(5)
            losers = movers.sort_values(by='Change (%)', ascending=True).head(5)
            with col_gain:
                st.success("🚀 Top Gainers")
                g_disp = gainers.copy()
                g_disp['Change (%)'] = g_disp['Change (%)'].map('{:+.2f}%'.format)
                g_disp['Price'] = g_disp['Price'].map('${:,.2f}'.format)
                st.table(g_disp)
            with col_loss:
                st.error("🔻 Top Losers")
                l_disp = losers.copy()
                l_disp['Change (%)'] = l_disp['Change (%)'].map('{:+.2f}%'.format)
                l_disp['Price'] = l_disp['Price'].map('${:,.2f}'.format)
                st.table(l_disp)
    except Exception as e:
        st.error(f"Error fetching market data: {e}")

# TAB 2: NEWS & INTELLIGENCE
with tab2:
    st.header("News & Trading Intelligence")
    sector_map = {
        "Aerospace & Defense": ['BA', 'LMT', 'RTX', 'NOC', 'GD'],
        "Future Tech & Speculative": ['ACHR', 'JOBY', 'RKLB', 'ASTS', 'SPCE'],
        "Technology": ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMD'],
        "Finance (Banking)": ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        "Energy": ['XOM', 'CVX', 'SHEL', 'BP', 'COP'],
        "Healthcare": ['LLY', 'JNJ', 'PFE', 'MRK', 'ABBV'],
        "Custom": [] 
    }
    col_sel, col_manual = st.columns([1, 2])
    with col_sel:
        selected_sector = st.selectbox("Select Sector", list(sector_map.keys()))
    if selected_sector == "Custom":
        with col_manual:
            tickers = st.multiselect("Select Tickers", ['BA', 'LMT', 'RTX', 'NOC', 'ACHR'], default=['BA'])
    else:
        tickers = sector_map[selected_sector]
        st.info(f"Scanning Top 5 in {selected_sector}: {', '.join(tickers)}")
    
    if st.button("Run Sector Sentiment & Trend Analysis"):
        st.divider()
        sentiment_summary = []
        all_news_rows = []
        progress_bar = st.progress(0)
        vader = SentimentIntensityAnalyzer()
        url_root = 'https://finviz.com/quote.ashx?t='
        
        try:
            # Get 3 months data to calculate bands and RSI
            tech_data = yf.download(tickers, period="3mo", progress=False)['Close']
        except:
            tech_data = pd.DataFrame()

        for i, ticker in enumerate(tickers):
            time.sleep(0.1) 
            progress_bar.progress((i + 1) / len(tickers))
            
            # --- 1. TECHNICAL STATS (RSI & BANDS) ---
            rsi_val = 50 
            buy_zone = "N/A"
            sell_zone = "N/A"
            current_price = 0
            
            try:
                t_series = None
                if not tech_data.empty:
                    if len(tickers) == 1 and isinstance(tech_data, pd.Series):
                        t_series = tech_data
                    elif ticker in tech_data.columns:
                        t_series = tech_data[ticker].dropna()
                
                if t_series is not None and not t_series.empty:
                    current_price = t_series.iloc[-1]
                    # RSI
                    rsi_series = calculate_rsi(t_series)
                    if not rsi_series.empty: rsi_val = rsi_series.iloc[-1]
                    
                    # Bollinger Bands for Ranges
                    upper, lower = calculate_bollinger_bands_val(t_series)
                    if not upper.empty and not lower.empty:
                        buy_zone = f"${lower.iloc[-1]:.2f}"
                        sell_zone = f"${upper.iloc[-1]:.2f}"
            except: pass

            # --- 2. NEWS SCRAPING ---
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = requests.get(url_root + ticker, headers=headers)
                html = BeautifulSoup(req.text, features='html.parser')
                news_table = html.find(id='news-table')
                if news_table:
                    ticker_headlines = []
                    for row in news_table.findAll('tr')[:10]:
                        if row.a is None: continue 
                        title = row.a.text
                        timestamp = row.td.text.split()
                        if len(timestamp) == 1: time_str = timestamp[0]; date_str = "Today" 
                        else: date_str = timestamp[0]; time_str = timestamp[1]
                        score = vader.polarity_scores(title)['compound']
                        ticker_headlines.append(score)
                        all_news_rows.append([ticker, date_str, time_str, title, score])
                    avg_score = np.mean(ticker_headlines) if ticker_headlines else 0
                    
                    sentiment_summary.append({
                        "Ticker": ticker,
                        "Price": current_price,
                        "Est. Buy Zone": buy_zone,
                        "Est. Sell Zone": sell_zone,
                        "RSI": rsi_val,
                        "Sentiment": avg_score
                    })
            except Exception as e: continue
        
        if sentiment_summary:
            summary_df = pd.DataFrame(sentiment_summary).sort_values(by="Sentiment", ascending=False)
            st.subheader(f"📊 Market Intelligence: {selected_sector}")
            display_df = summary_df.copy()
            
            def get_sentiment_label(score): return "Bullish 🟢" if score > 0.1 else "Bearish 🔴" if score < -0.1 else "Neutral ⚪"
            def get_rsi_signal(rsi): return "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral ⚪"

            display_df['News Verdict'] = display_df['Sentiment'].apply(get_sentiment_label)
            display_df['Status'] = display_df['RSI'].apply(get_rsi_signal)
            
            display_df['Price'] = display_df['Price'].map('${:,.2f}'.format)
            display_df['Sentiment'] = display_df['Sentiment'].map('{:+.3f}'.format)
            display_df['RSI'] = display_df['RSI'].map('{:.1f}'.format)
            
            # Reorder columns for readability
            cols = ["Ticker", "Price", "Status", "Est. Buy Zone", "Est. Sell Zone", "News Verdict", "Sentiment", "RSI"]
            st.dataframe(display_df[cols].set_index("Ticker"), use_container_width=True)
        
        if all_news_rows:
            st.subheader("📰 Live News Feed")
            news_df = pd.DataFrame(all_news_rows, columns=['Ticker', 'Date', 'Time', 'Headline', 'Score'])
            def color_sentiment(val): return f'color: {"red" if val < -0.1 else "green" if val > 0.1 else "black"}'
            st.dataframe(news_df.style.map(color_sentiment, subset=['Score']), use_container_width=True, height=500)

# TAB 3: PRO CHARTING & TRADING
with tab3:
    with st.sidebar:
        st.header("Terminal Settings")
        sim_ticker = st.text_input("Ticker Symbol", value="BA")
        st.subheader("📋 Fundamental Health")
        fund_data = get_fundamentals(sim_ticker)
        if fund_data:
            c1, c2 = st.columns(2)
            c1.metric("Mkt Cap", f"{fund_data['Market Cap']}", help="Total value")
            c2.metric("P/E Ratio", f"{fund_data['P/E Ratio']}", help="Price to Earnings")
            c1.metric("Div Yield", f"{fund_data['Dividend Yield']}", help="Annual Dividend %")
            c2.metric("Profit Margin", f"{fund_data['Profit Margin']}", help="Net Income / Revenue")
        else: st.caption("Data unavailable")
        st.subheader("Timeframe Settings")
        c1, c2 = st.columns(2)
        interval_map = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "30 Minutes": "30m", "1 Hour": "1h", "1 Day": "1d", "1 Week": "1wk"}
        interval_label = c1.selectbox("Interval", list(interval_map.keys()), index=5)
        selected_interval = interval_map[interval_label]
        period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        selected_period = c2.selectbox("Lookback", period_options, index=5)
        st.divider()
        st.subheader("Chart Style")
        chart_style = st.selectbox("Type", ["Line Chart", "Candlestick"], index=1)
        st.subheader("Pattern Recognition")
        show_candles = st.checkbox("Show All Micro Patterns (Candles)", value=True)
        show_macro = st.checkbox("Show All Macro Patterns (Chart)", value=True)
        st.divider()
        st.subheader("Prediction Engine")
        enable_prediction = st.checkbox("Enable Monte Carlo Sim", value=False)
        verify_backtest = st.checkbox("Verify Accuracy (Backtest)", value=False)
        drift_option = st.selectbox("Trend Model", ["Historical Average (GBM)", "Linear Regression Trend"], help="Choose how the simulation calculates the future path.")
        forecast_days = st.slider("Forecast/Backtest Horizon (Days)", min_value=10, max_value=252, value=30)
        vol_scale = st.slider("Volatility Scaler", 0.1, 1.0, 0.5)
        sim_count = st.slider("Simulations", 100, 1000, 200)
        st.divider()
        st.subheader("Overlays")
        macro_overlay = st.checkbox("Overlay Macro Data", value=False)
        if macro_overlay:
            macro_asset = st.selectbox("Select Macro Asset", ["^TNX (10Y Yield)", "CL=F (Crude Oil)"])
        overlays = st.multiselect("Select Indicators", ["50 SMA", "200 SMA", "Bollinger Bands", "Fibonacci", "Ichimoku Cloud", "VWAP", "Support/Resistance"], default=["50 SMA"])
        sub_chart = st.selectbox("Bottom Panel", ["Volume", "RSI", "MACD", "Stochastic", "OBV", "None"], index=0)
        strategy_type = st.selectbox("Strategy", ["Mean Reversion (MA50)", "Golden Cross", "RSI Reversal"])

    st.header(f"{sim_ticker} Technical Analysis ({interval_label})")
    try:
        data = get_stock_data(sim_ticker, selected_period, selected_interval)
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()
    if len(data) > 0:
        data = calculate_indicators(data)
        data = identify_candlestick_patterns(data)
        data = identify_macro_patterns(data)
        data['Signal'] = 0
        if strategy_type == "Mean Reversion (MA50)": data.loc[data['Close'] < data['MA50'], 'Signal'] = 1
        elif strategy_type == "Golden Cross": data.loc[data['MA50'] > data['MA200'], 'Signal'] = 1
        elif strategy_type == "RSI Reversal": data.loc[data['RSI'] < 30, 'Signal'] = 1
        data['Buy_Trade'] = (data['Signal'] == 1) & (data['Signal'].shift(1) == 0)
        data['Sell_Trade'] = (data['Signal'] == 0) & (data['Signal'].shift(1) == 1)
        sharpe, max_dd = calculate_quant_metrics(data)
        st.markdown(f"""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
            <span style="margin-right: 20px;"><b>🛡️ Quantitative Risk Metrics:</b></span>
            <span style="margin-right: 20px;">Sharpe Ratio: <b>{sharpe:.2f}</b></span>
            <span>Max Drawdown: <span style="color: red;"><b>{max_dd:.2f}%</b></span></span>
        </div>
        """, unsafe_allow_html=True)
        nrows = 2 if sub_chart != "None" else 1
        height_ratios = [3, 1] if sub_chart != "None" else [1]
        fig, ax = plt.subplots(nrows, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': height_ratios})
        if nrows == 1: ax_main = ax
        else: ax_main, ax_sub = ax
        
        if macro_overlay:
            macro_ticker = macro_asset.split()[0]
            try:
                macro_data = yf.download(macro_ticker, period=selected_period, interval=selected_interval, progress=False)['Close']
                ax2 = ax_main.twinx()
                ax2.plot(macro_data.index, macro_data, color='gray', alpha=0.3, linestyle='-', linewidth=1, label=macro_asset)
                ax2.set_ylabel(macro_asset, color='gray')
            except: pass

        if chart_style == "Line Chart":
            ax_main.plot(data['Date'], data['Close'], label='Price', color='black', alpha=0.6)
        else:
            ax_main.vlines(data['Date'], data['Low'], data['High'], color='gray', linewidth=0.8, alpha=0.8)
            up = data[data['Close'] >= data['Open']]
            down = data[data['Close'] < data['Open']]
            width = (data['Date'].iloc[1] - data['Date'].iloc[0]).total_seconds() / 86400 * 0.6 if len(data) > 1 else 0.5
            ax_main.bar(up['Date'], up['Close'] - up['Open'], bottom=up['Open'], color='green', width=width, align='center')
            ax_main.bar(down['Date'], down['Close'] - down['Open'], bottom=down['Open'], color='red', width=width, align='center')
        
        if show_candles:
            pat = data[data['Pattern_Bullish_Engulfing']]
            ax_main.scatter(pat['Date'], pat['Low'] * 0.99, marker='^', color='blue', s=100, label='Bull Engulf', zorder=10)
            pat = data[data['Pattern_Bearish_Engulfing']]
            ax_main.scatter(pat['Date'], pat['High'] * 1.01, marker='v', color='purple', s=100, label='Bear Engulf', zorder=10)
            pat = data[data['Pattern_Hammer']]
            ax_main.scatter(pat['Date'], pat['Low'] * 0.99, marker='*', color='gold', s=100, label='Hammer', zorder=10)
            pat = data[data['Pattern_Doji']]
            ax_main.scatter(pat['Date'], pat['High'] * 1.01, marker='x', color='black', s=50, label='Doji', zorder=10)
        
        if show_macro:
            pat = data[data['Pattern_HeadShoulders']]
            if not pat.empty: ax_main.scatter(pat['Date'], pat['High']*1.02, marker='v', color='red', s=150, label='H&S', zorder=20)
            pat = data[data['Pattern_InvHeadShoulders']]
            if not pat.empty: ax_main.scatter(pat['Date'], pat['Low']*0.98, marker='^', color='lime', s=150, label='Inv H&S', zorder=20)
            pat = data[data['Pattern_DoubleTop']]
            if not pat.empty: ax_main.scatter(pat['Date'], pat['High'], marker='x', color='orange', s=120, label='Double Top', zorder=20)
            pat = data[data['Pattern_Wedge']]
            if not pat.empty: ax_main.scatter(pat['Date'], pat['Close'], marker='o', color='cyan', s=100, label='Wedge', zorder=20)

        if "50 SMA" in overlays: ax_main.plot(data['Date'], data['MA50'], color='blue', linestyle='--', label='50 SMA')
        if "200 SMA" in overlays: ax_main.plot(data['Date'], data['MA200'], color='orange', linestyle='--', label='200 SMA')
        if "Bollinger Bands" in overlays:
            ax_main.plot(data['Date'], data['BB_Upper'], color='green', alpha=0.1)
            ax_main.plot(data['Date'], data['BB_Lower'], color='red', alpha=0.1)
            ax_main.fill_between(data['Date'], data['BB_Upper'], data['BB_Lower'], color='gray', alpha=0.1)
        if "Fibonacci" in overlays:
            max_p = data['Close'].max(); min_p = data['Close'].min(); diff = max_p - min_p
            for lvl, c in zip([0, 0.236, 0.382, 0.5, 0.618, 1], ['gray', 'red', 'orange', 'blue', 'green', 'gray']):
                ax_main.axhline(max_p - (diff * lvl), linestyle=':', alpha=0.5, color=c)
        if "Ichimoku Cloud" in overlays:
            ax_main.plot(data['Date'], data['Tenkan'], color='red', linewidth=1); ax_main.plot(data['Date'], data['Kijun'], color='blue', linewidth=1)
            ax_main.fill_between(data['Date'], data['SpanA'], data['SpanB'], where=data['SpanA']>=data['SpanB'], color='lightgreen', alpha=0.3)
            ax_main.fill_between(data['Date'], data['SpanA'], data['SpanB'], where=data['SpanA']<data['SpanB'], color='lightcoral', alpha=0.3)
        if "VWAP" in overlays: ax_main.plot(data['Date'], data['VWAP'], color='purple', linewidth=1.5, label='VWAP')
        if "Support/Resistance" in overlays:
            sr_levels = find_support_resistance(data)
            for lvl in sr_levels: ax_main.axhline(lvl, color='gray', linestyle='-.', alpha=0.6, linewidth=1)
        
        if enable_prediction:
            is_backtest = verify_backtest
            f_dates, paths, actual_path, linear_line, lower_path, upper_path = monte_carlo_simulation(sim_ticker, forecast_days, sim_count, vol_scale, backtest=is_backtest, drift_method=drift_option)
            if paths is not None:
                ax_main.fill_between(f_dates, lower_path, upper_path, color='red', alpha=0.1, label='90% Confidence Cloud')
                median_path = np.median(paths, axis=1)
                label_prefix = "Backtest" if is_backtest else "Forecast"
                ax_main.plot(f_dates, median_path, color='red', linestyle='--', linewidth=2, label=f'{label_prefix} Median')
                if linear_line is not None:
                    ax_main.plot(f_dates, linear_line, color='green', linestyle=':', linewidth=2, label='Linear Trend')
                
                # Advanced Accuracy Check
                if is_backtest and actual_path is not None:
                    ax_main.plot(f_dates, [actual_path.iloc[0]] + actual_path.tolist(), color='blue', linewidth=3, label='ACTUAL Price')
                    
                    min_len = min(len(actual_path), len(lower_path[1:]))
                    actual_trimmed = actual_path.values[:min_len]
                    lower_trimmed = lower_path[1:min_len+1]
                    upper_trimmed = upper_path[1:min_len+1]
                    
                    inside_cloud = (actual_trimmed >= lower_trimmed) & (actual_trimmed <= upper_trimmed)
                    accuracy_pct = np.mean(inside_cloud) * 100
                    
                    if accuracy_pct > 80:
                         st.success(f"✅ HIGH ACCURACY: Actual price stayed within the 90% confidence cloud **{accuracy_pct:.1f}%** of the time.")
                    elif accuracy_pct > 50:
                         st.warning(f"⚠️ MODERATE ACCURACY: Actual price stayed within the cloud **{accuracy_pct:.1f}%** of the time.")
                    else:
                         st.error(f"❌ LOW ACCURACY: Actual price only stayed within the cloud **{accuracy_pct:.1f}%** of the time. Consider adjusting volatility.")

                final_prices = paths[-1, :]
                m1, m2, m3 = st.columns(3)
                m1.metric("Bear Case (5%)", f"${np.percentile(final_prices, 5):.2f}")
                m2.metric(f"{label_prefix} Median", f"${np.median(final_prices):.2f}")
                m3.metric("Bull Case (95%)", f"${np.percentile(final_prices, 95):.2f}")
        
        buys = data[data['Buy_Trade']]; sells = data[data['Sell_Trade']]
        ax_main.scatter(buys['Date'], buys['Close'], marker='^', color='green', s=100, zorder=5)
        ax_main.scatter(sells['Date'], sells['Close'], marker='v', color='red', s=100, zorder=5)
        if chart_style == "Line Chart" or overlays or show_candles or show_macro: ax_main.legend(loc='upper left')
        if sub_chart != "None":
            if sub_chart == "RSI": ax_sub.plot(data['Date'], data['RSI'], color='purple'); ax_sub.axhline(70, c='red', ls='--'); ax_sub.axhline(30, c='green', ls='--')
            elif sub_chart == "MACD": ax_sub.plot(data['Date'], data['MACD_Line'], c='blue'); ax_sub.plot(data['Date'], data['MACD_Signal'], c='orange'); ax_sub.bar(data['Date'], data['MACD_Line'] - data['MACD_Signal'], color='gray', alpha=0.3)
            elif sub_chart == "Stochastic": ax_sub.plot(data['Date'], data['Stoch_K'], color='blue', label='%K'); ax_sub.plot(data['Date'], data['Stoch_D'], color='orange', label='%D'); ax_sub.axhline(80, c='red', ls='--'); ax_sub.axhline(20, c='green', ls='--'); ax_sub.legend()
            elif sub_chart == "Volume": ax_sub.bar(data['Date'], data['Volume'], color='black')
            elif sub_chart == "OBV": ax_sub.plot(data['Date'], data['OBV'], color='teal')
        st.pyplot(fig)
        latest = data.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Close", f"${latest['Close']:.2f}")
        c2.metric("RSI", f"{latest['RSI']:.2f}" if not pd.isna(latest['RSI']) else "N/A")
        decision = "BUY" if latest['Signal'] == 1 else "SELL"
        c3.metric("Algo Action", decision, delta_color="normal" if decision=="SELL" else "inverse")
        csv = data.to_csv().encode('utf-8')
        st.download_button(label="📥 Download Chart Data (CSV)", data=csv, file_name=f"{sim_ticker}_data.csv", mime="text/csv")
    else: st.error("No data found. If using 1-Minute interval, reduce Lookback to '5d'.")

# TAB 4: SECTOR RESEARCH
with tab4:
    st.header("🔍 Automated Sector Scanner")
    sector_dict = {
        "Aerospace & Defense": ['BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'HII', 'TDG', 'TXT', 'AIR.PA'],
        "Future Tech & Speculative": ['ACHR', 'JOBY', 'RKLB', 'ASTS', 'SPCE', 'LUNR', 'EH', 'PLTR', 'QS', 'IONQ'],
        "Technology": ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'QCOM'],
        "Energy (Oil & Gas)": ['XOM', 'CVX', 'SHEL', 'TTE', 'BP', 'COP', 'SLB', 'EOG', 'PXD', 'VLO'],
        "Finance (Banking)": ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA'],
        "Healthcare (Pharma)": ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'NVS', 'AZN', 'AMGN']
    }
    selected_sector_scan = st.selectbox("Select Sector", list(sector_dict.keys()), key="sector_scan")
    
    # NEW FILTER OPTIONS
    filter_option = st.radio("Filter By:", ["All", "Upcoming Earnings (30d)", "High Dividend (>3%)"], horizontal=True)
    
    if st.button("🚀 Run Sector Simulation"):
        tickers = sector_dict[selected_sector_scan]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Pre-fetch for filters
        try:
            info_batch = {}
            # We can't batch fetch info well with yfinance, so we do it in loop or use safe fallback
        except: pass

        filtered_tickers = []
        
        # FILTERING LOGIC
        if filter_option == "All":
            filtered_tickers = tickers
        else:
            for t in tickers:
                try:
                    tick = yf.Ticker(t)
                    if filter_option == "High Dividend (>3%)":
                        div = tick.info.get('dividendYield', 0)
                        if div and div > 0.03: filtered_tickers.append(t)
                    elif filter_option == "Upcoming Earnings (30d)":
                        # Earnings calendar is notoriously flaky in free APIs
                        # We use a placeholder logic: if data exists and is soon
                        cal = tick.calendar
                        if cal is not None and not cal.empty:
                            # Check date (simplified for stability)
                            filtered_tickers.append(t)
                except: pass
            
            # Fallback if filter returns empty (to show something)
            if not filtered_tickers:
                st.warning("No stocks matched the filter. Showing all.")
                filtered_tickers = tickers

        for i, t in enumerate(filtered_tickers):
            status_text.text(f"Simulating {t}...")
            _, paths, _, _, _, _ = monte_carlo_simulation(t, 30, 200, vol_scale=0.8, drift_method="Historical")
            if paths is not None:
                current_price = paths[0, 0]
                median_future = np.median(paths[-1, :])
                p_return = ((median_future - current_price) / current_price) * 100
                results.append({"Ticker": t, "Current Price": current_price, "Predicted Price (30d)": median_future, "Exp. Return (%)": p_return})
            progress_bar.progress((i + 1) / len(filtered_tickers))
        status_text.text("Scan Complete!")
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Exp. Return (%)", ascending=False).reset_index(drop=True)
            st.subheader(f"Top Opportunities in {selected_sector_scan}")
            top_cols = st.columns(3)
            for i in range(min(3, len(res_df))):
                row = res_df.iloc[i]
                top_cols[i].metric(label=f"🥇 Rank {i+1}: {row['Ticker']}", value=f"${row['Predicted Price (30d)']:.2f}", delta=f"{row['Exp. Return (%)']:.2f}%")
            st.dataframe(res_df.style.format({"Current Price": "${:.2f}", "Predicted Price (30d)": "${:.2f}", "Exp. Return (%)": "{:+.2f}%"}).background_gradient(subset=["Exp. Return (%)"], cmap="RdYlGn"), use_container_width=True)
            scan_csv = res_df.to_csv().encode('utf-8')
            st.download_button(label="📥 Download Scan Results (CSV)", data=scan_csv, file_name=f"{selected_sector_scan}_scan.csv", mime="text/csv")
        else:
            st.warning("No data returned from simulation.")

# TAB 5: COMPETITOR RESEARCH
with tab5:
    st.header("🐋 Whale & Competitor Tracker")
    st.subheader("🏛️ Congressional Trading (Live)")
    st.caption("Recent trades by US Congress members (Source: Capitol Trades)")
    components.iframe("https://www.capitoltrades.com/trades", height=600, scrolling=True)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏰 Warren Buffett (Berkshire)")
        st.caption("Top holdings (Source: Dataroma)")
        try:
            buffett_data = get_buffett_portfolio()
            if not buffett_data.empty: st.dataframe(buffett_data.head(15), use_container_width=True, height=500)
            else: st.info("Could not fetch Buffett data.")
        except Exception as e: st.error(f"Error: {e}")
    with col2:
        st.subheader("🕵️ Corporate Insider Trading")
        st.caption("CEO/Director Trades (Source: Finviz)")
        try:
            insider_data = get_insider_trading()
            if not insider_data.empty:
                def color_transaction(val): return f'color: {"green" if "Buy" in val else "red" if "Sell" in val else "black"}'
                st.dataframe(insider_data.style.map(color_transaction, subset=['Transaction']), use_container_width=True, height=500)
            else: st.info("Could not fetch Insider data.")
        except Exception as e: st.error(f"Error: {e}")

# TAB 6: PORTFOLIO ANALYSIS
with tab6:
    st.header("📊 Quantitative Portfolio Analysis")
    default_tickers = ['BA', 'LMT', 'RTX', 'NOC', '^GSPC']
    comparison_tickers = st.multiselect("Select Assets to Compare", ['BA', 'LMT', 'RTX', 'NOC', 'GD', 'ACHR', 'JOBY', 'NVDA', '^GSPC', '^DJI'], default=default_tickers)
    comp_period = st.selectbox("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    if st.button("Run Portfolio Analysis"):
        st.divider()
        try:
            batch_data = yf.download(comparison_tickers, period=comp_period, progress=False)['Close']
            st.subheader("🏎️ Relative Performance (Rebased to 0%)")
            normalized_data = (batch_data / batch_data.iloc[0]) - 1
            fig_race, ax_race = plt.subplots(figsize=(14, 7))
            for col in normalized_data.columns:
                linewidth = 3 if col == '^GSPC' else 1.5
                linestyle = '--' if col == '^GSPC' else '-'
                ax_race.plot(normalized_data.index, normalized_data[col], label=col, linewidth=linewidth, linestyle=linestyle)
            ax_race.axhline(0, color='black', linewidth=1)
            ax_race.set_ylabel("Return (%)")
            ax_race.legend()
            ax_race.grid(True, which='both', linestyle='--', linewidth=0.5)
            vals = ax_race.get_yticks()
            ax_race.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
            st.pyplot(fig_race)
            st.subheader("🔗 Correlation Matrix (Diversification Check)")
            
            corr_matrix = batch_data.pct_change().corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax_corr)
            st.pyplot(fig_corr)
        except Exception as e: st.error(f"Error running analysis: {e}")

# TAB 7: VALUATION & RISK
with tab7:
    st.header("💰 Valuation & Risk Engineering")
    st.subheader("1. Interactive DCF Valuation")
    col_dcf_1, col_dcf_2 = st.columns(2)
    if fund_data:
        current_price = yf.Ticker(sim_ticker).history(period='1d')['Close'].iloc[-1]
        fcf = fund_data.get('Free Cash Flow')
        shares = fund_data.get('Shares Outstanding')
        if fcf and shares:
            fcf_per_share = fcf / shares
            metric_used = "Free Cash Flow"
        else:
            try:
                fcf_per_share = yf.Ticker(sim_ticker).info.get('trailingEps', 0)
                metric_used = "Earnings Per Share (EPS Proxy)"
            except:
                fcf_per_share = 0
                metric_used = "N/A"
        with col_dcf_1:
            st.info(f"Base Metric ({metric_used}): ${fcf_per_share:.2f}")
            growth_rate = st.slider("Expected Growth Rate (%)", 0, 50, 10) / 100
            discount_rate = st.slider("Discount Rate (WACC) (%)", 5, 20, 9) / 100
            terminal_multiple = st.slider("Terminal Multiple (Exit P/E)", 5, 30, 15)
        with col_dcf_2:
            future_fcf = []
            for i in range(1, 6): future_fcf.append(fcf_per_share * ((1 + growth_rate) ** i))
            terminal_value = future_fcf[-1] * terminal_multiple
            dcf_value = 0
            for i in range(5): dcf_value += future_fcf[i] / ((1 + discount_rate) ** (i + 1))
            dcf_value += terminal_value / ((1 + discount_rate) ** 5)
            delta_val = dcf_value - current_price
            st.metric("Intrinsic Value (Fair Price)", f"${dcf_value:.2f}", delta=f"{delta_val:.2f}")
            if dcf_value > current_price: st.success("✅ UNDERVALUED: Trading below intrinsic value.")
            else: st.error("⚠️ OVERVALUED: Trading above intrinsic value.")
    st.divider()
    st.subheader("2. Value at Risk (VaR) Analysis")
    try:
        hist_data = yf.download(sim_ticker, period="1y", progress=False)['Close']
        daily_returns = hist_data.pct_change().dropna()
        var_95 = np.percentile(daily_returns, 5)
        fig_var, ax_var = plt.subplots(figsize=(10, 5))
        sns.histplot(daily_returns, bins=50, kde=True, color="blue", ax=ax_var)
        ax_var.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.2%}')
        ax_var.legend(); st.pyplot(fig_var)
        st.error(f"📉 **95% Value at Risk:** {var_95:.2%}")
        st.markdown(f"This means that on any given day, there is a **95% chance** that {sim_ticker} will NOT drop more than **{abs(var_95*100):.2f}%**.")
    except Exception as e: st.error("Insufficient data for VaR analysis.")