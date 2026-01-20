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
from scipy.stats import linregress
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroQuant Pro Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin-bottom: 0px; }
    .ticker-text { font-size: 32px; font-weight: 700; color: #000; margin-right: 15px; }
    .price-text { font-size: 42px; font-weight: 800; color: #000; }
    .delta-pos { color: #00873c; font-size: 24px; font-weight: 600; }
    .delta-neg { color: #eb0f29; font-size: 24px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✈️ Aerospace Quantitative Terminal")

nltk.download('vader_lexicon', quiet=True)

# ==========================================
# 1. DATA FETCHING (SHARED)
# ==========================================

@st.cache_data(ttl=2)
def get_stock_data(ticker, period="max", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'index' in data.columns:
            data.rename(columns={'index': 'Date'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=2)
def get_live_price_data(ticker):
    try:
        info = yf.Ticker(ticker).info
        current = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price', 0)
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose', 0)
        post_market = info.get('postMarketPrice')
        return current, prev_close, post_market
    except: return 0, 0, None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Name": info.get("shortName", ticker),
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "Free Cash Flow": info.get("freeCashflow", None),
            "Shares Outstanding": info.get("sharesOutstanding", None)
        }
    except: return None

@st.cache_data(ttl=300)
def get_market_overview():
    market_tickers = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225', 'GBPUSD=X', 'EURUSD=X', 'JPY=X', 'GC=F', 'CL=F']
    try:
        market_data = yf.download(market_tickers, period="5d", progress=False)
        if isinstance(market_data.columns, pd.MultiIndex): closes = market_data['Close']
        else: closes = market_data['Close'] if 'Close' in market_data else market_data
        closes = closes.ffill().dropna()
        
        if len(closes) >= 2:
            latest = closes.iloc[-1]
            prev = closes.iloc[-2]
            change_pct = ((latest - prev) / prev) * 100
            
            stock_universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AMD', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'AIR.PA', 'RR.L', 'JPM', 'BAC', 'GS', 'HSBC', 'XOM', 'CVX', 'SHEL', 'PFE', 'LLY', 'JNJ', 'F', 'GM', 'TM', 'PLTR', 'COIN', 'HOOD', 'DIS', 'NFLX']
            stock_raw = yf.download(stock_universe, period="5d", progress=False)
            if isinstance(stock_raw.columns, pd.MultiIndex): stock_closes = stock_raw['Close'].ffill().dropna()
            else: stock_closes = stock_raw.ffill().dropna()
            
            s_latest = stock_closes.iloc[-1]
            s_prev = stock_closes.iloc[-2]
            s_change = ((s_latest - s_prev) / s_prev) * 100
            
            movers_df = pd.DataFrame({'Price': s_latest, 'Change (%)': s_change})
            return latest, change_pct, movers_df
        return pd.Series(), pd.Series(), pd.DataFrame()
    except: return pd.Series(), pd.Series(), pd.DataFrame()

@st.cache_data(ttl=3600)
def get_insider_trading():
    try:
        url = 'https://finviz.com/insidertrading.ashx'
        headers = {'User-Agent': 'Mozilla/5.0'}
        dfs = pd.read_html(requests.get(url, headers=headers).text)
        for df in dfs:
            if 'Owner' in df.columns: return df.head(10)[['Ticker', 'Owner', 'Relation', 'Date', 'Transaction', 'Cost', 'Shares']]
            elif len(df)>1 and 'Owner' in str(df.iloc[0]): 
                df.columns = df.iloc[0]; return df[1:].head(10)[['Ticker', 'Owner', 'Relation', 'Date', 'Transaction', 'Cost', 'Shares']]
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_buffett_portfolio():
    try:
        url = 'https://www.dataroma.com/m/holdings.php?m=BRK'
        headers = {'User-Agent': 'Mozilla/5.0'}
        dfs = pd.read_html(requests.get(url, headers=headers).text)
        for df in dfs:
            if 'Stock' in df.columns: return df[['Ticker', 'Stock', '% of Portfolio']].head(15)
        return pd.DataFrame()
    except: return pd.DataFrame()

# ==========================================
# 2. AI & GEOPOLITICS
# ==========================================

@st.cache_data(ttl=300)
def analyze_geopolitics(ticker):
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers)
        html = BeautifulSoup(req.text, features='html.parser')
        news_table = html.find(id='news-table')
        
        if not news_table: return 0.0, 0.0, "No news data available."
        
        vader = SentimentIntensityAnalyzer()
        risk_keywords = ['war', 'conflict', 'sanctions', 'tension', 'military', 'greenland', 'arctic', 'rare earth', 'china', 'tariffs', 'trade war', 'embargo']
        
        risk_hits = 0; sentiment_scores = []
        for row in news_table.findAll('tr')[:20]: 
            if row.a:
                text = row.a.text.lower()
                if any(word in text for word in risk_keywords): risk_hits += 1
                sentiment_scores.append(vader.polarity_scores(text)['compound'])
        
        risk_factor = min(risk_hits * 0.3, 1.0) 
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        status = "CRITICAL TENSION" if risk_factor > 0.5 else "Elevated Risk" if risk_factor > 0.1 else "Stable"
        
        return risk_factor, avg_sentiment, f"Geopolitical Status: {status} ({risk_hits} flags). Sentiment: {avg_sentiment:.2f}."
    except: return 0.0, 0.0, "Error analyzing news."

# ==========================================
# 3. PATTERN RECOGNITION
# ==========================================

def detect_comprehensive_patterns(data):
    df = data.copy()
    df['Body'] = df['Close'] - df['Open']
    df['AbsBody'] = df['Body'].abs()
    df['Range'] = df['High'] - df['Low']
    df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    patterns = {}
    patterns['Doji'] = df['AbsBody'] <= (0.05 * df['Range'])
    patterns['Dragonfly Doji'] = patterns['Doji'] & (df['LowerWick'] > 0.6 * df['Range'])
    hammer_shape = (df['LowerWick'] >= 2 * df['AbsBody']) & (df['UpperWick'] <= 0.2 * df['AbsBody'])
    patterns['Hammer'] = hammer_shape & (df['Close'].shift(1) < df['Open'].shift(1))
    patterns['Hanging Man'] = hammer_shape & (df['Close'].shift(1) > df['Open'].shift(1))
    patterns['Bullish Engulfing'] = (df['Body'] > 0) & (df['Body'].shift(1) < 0) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    patterns['Bearish Engulfing'] = (df['Body'] < 0) & (df['Body'].shift(1) > 0) & (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))
    
    for name, mask in patterns.items():
        df[f'Pat_{name}'] = mask

    return df, list(patterns.keys())

def calculate_pattern_accuracy(df, pattern_cols):
    stats = []
    for pat in pattern_cols:
        col_name = f'Pat_{pat}'
        if col_name not in df.columns: continue
        indices = df.index[df[col_name] == True]
        if len(indices) > 0:
            wins = 0; total = 0
            for idx in indices:
                loc = df.index.get_loc(idx)
                if loc + 3 < len(df):
                    future = df['Close'].iloc[loc + 3]; curr = df['Close'].iloc[loc]
                    is_bull = any(x in pat for x in ['Bull', 'Morning', 'White', 'Hammer', 'Dragonfly'])
                    if is_bull and future > curr: wins += 1
                    elif not is_bull and future < curr: wins += 1
                    total += 1
            if total > 0:
                acc = (wins / total) * 100
                last_idx = indices[-1]
                last_date = str(last_idx.strftime('%Y-%m-%d')) if hasattr(last_idx, 'strftime') else str(last_idx)
                stats.append({"Pattern": pat, "Reliability": f"{acc:.1f}%", "Count": total, "Last Seen": last_date})
    return pd.DataFrame(stats).sort_values("Last Seen", ascending=False)

def calculate_smart_trendline(data):
    df = data.copy().dropna()
    if len(df) < 5: return None, None
    x = np.arange(len(df)); y = df['Close'].values
    slope, intercept, _, _, _ = linregress(x, y)
    line_y = slope * x + intercept
    return df['Date'], line_y

def calculate_indicators(data):
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    mid = data['Close'].rolling(20).mean(); std = data['Close'].rolling(20).std()
    data['BB_Upper'] = mid + (2 * std); data['BB_Lower'] = mid - (2 * std)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain/loss
    data['RSI'] = 100 - (100/(1+rs))
    data['MACD_Line'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9).mean()
    v = data['Volume'].values; tp = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (tp * v).cumsum() / v.cumsum()
    return data

def monte_carlo_simulation(ticker, days_ahead, simulations=200, vol_scale=1.0, backtest=False, drift_opt="Historical", use_ai=False):
    try:
        sim_data = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(sim_data.columns, pd.MultiIndex): sim_data.columns = sim_data.columns.get_level_values(0)
        
        if backtest:
            actual = sim_data['Close'].iloc[-days_ahead:]; train = sim_data.iloc[:-days_ahead]
        else:
            actual = None; train = sim_data
            
        log_ret = np.log(1 + train['Close'].pct_change()).dropna()
        risk_fac = 0; sentiment_adj = 0
        if use_ai: risk_fac, sentiment_adj, _ = analyze_geopolitics(ticker)
        
        vol = (0.8*log_ret.tail(20).std() + 0.2*log_ret.std()) * vol_scale
        if risk_fac > 0.1: vol *= (1 + risk_fac)
        
        if drift_opt == "Linear Regression Trend":
            x = np.arange(len(train)); y = np.log(train['Close'])
            slope, _, _, _, _ = linregress(x, y)
            mu = slope
        else: mu = log_ret.mean()
            
        mu = mu + (sentiment_adj * vol * 0.1) 
        drift = mu - (0.5 * vol**2)
        start = train['Close'].iloc[-1]
        paths = np.zeros((days_ahead+1, simulations)); paths[0] = start
        
        for t in range(1, days_ahead+1):
            shock = np.random.normal(0, 1, simulations)
            paths[t] = paths[t-1] * np.exp(drift + vol * shock)
            
        dates = [train.index[-1] + timedelta(days=i) for i in range(days_ahead+1)]
        low = np.percentile(paths, 5, axis=1); high = np.percentile(paths, 95, axis=1)
        return dates, paths, actual, low, high, risk_fac, None
    except: return None, None, None, None, None, 0, None

def stylized_price_display(price, prev_close):
    if prev_close is None or prev_close == 0:
        delta_abs = 0; delta_pct = 0; color_class = ""; sign = ""
    else:
        delta_abs = price - prev_close
        delta_pct = (delta_abs / prev_close) * 100
        color_class = "delta-pos" if delta_abs >= 0 else "delta-neg"
        sign = "+" if delta_abs >= 0 else ""

    return f"""
    <div class="price-container">
        <span class="price-text">{price:,.2f}</span>
        <span class="{color_class}" style="margin-left: 15px;">{sign}{delta_abs:.2f} ({sign}{delta_pct:.2f}%)</span>
    </div>
    """

def calculate_quant_metrics(data):
    if len(data) < 2: return 0, 0
    returns = data['Close'].pct_change().dropna()
    sharpe = (returns.mean() * 252) / (returns.std() * (252**0.5)) if returns.std() != 0 else 0
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return sharpe, drawdown.min() * 100

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands_val(data, window=20):
    mid = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = mid + (2 * std)
    lower = mid - (2 * std)
    return upper, lower

# ==========================================
# 5. DASHBOARD UI
# ==========================================

with st.sidebar:
    st.header("Terminal Settings")
    sim_ticker = st.text_input("Ticker Symbol", value="BA").upper()
    
    live_mode = st.checkbox("⚡ Live Mode (2s)", value=False)
    if live_mode: time.sleep(2); st.rerun()

    fund_data = get_fundamentals(sim_ticker)
    name_display = fund_data.get('Name', sim_ticker) if fund_data else sim_ticker
    
    st.subheader(f"{sim_ticker}")
    curr, prev, post = get_live_price_data(sim_ticker)
    st.markdown(stylized_price_display(curr, prev), unsafe_allow_html=True)
    
    if post and post != curr:
        p_delta = ((post - curr)/curr)*100
        st.metric("After Hours", f"${post:,.2f}", f"{p_delta:+.2f}%", delta_color="off")

    st.markdown("---")
    
    if fund_data:
        c1, c2 = st.columns(2)
        c1.metric("Mkt Cap", f"{fund_data['Market Cap']}", help="Total value")
        c2.metric("P/E Ratio", f"{fund_data['P/E Ratio']}", help="Price to Earnings")
        c1.metric("Div Yield", f"{fund_data['Dividend Yield']}", help="Annual Dividend %")
        c2.metric("Profit Margin", f"{fund_data['Profit Margin']}", help="Net Income / Revenue")
    
    st.divider()
    st.subheader("Analysis Settings")
    chart_style = st.selectbox("Chart Type", ["Candlestick", "Line Chart"])
    days = st.slider("Forecast Days", 1, 365, 30)
    sims = st.slider("Simulations", 100, 1000, 200)
    overlays = st.multiselect("Indicators", ["50 SMA", "200 SMA", "Bollinger", "VWAP", "Trend Channel"], default=["50 SMA"])
    sub_chart = st.selectbox("Sub-Chart", ["Volume", "RSI", "MACD", "None"], index=0)
    
    st.subheader("AI Engine")
    use_ai = st.checkbox("Auto-Adjust for Geopolitics", value=True)
    backtest = st.checkbox("Backtest Mode", value=False)
    vol_scale = st.slider("Volatility Scaler", 0.1, 1.0, 0.5)
    drift_option = st.selectbox("Trend Model", ["Historical Average (GBM)", "Linear Regression Trend"])
    show_candles = st.checkbox("Show Patterns", value=True)

# --- TAB LOGIC ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Market Overview", "📰 News & Intelligence", "📈 Pro Charting", 
    "🔍 Sector Research", "🐋 Competitor Research", "📊 Portfolio Analysis", "💰 Valuation & Risk"
])

# TAB 1: MARKETS
with tab1:
    st.header("Global Market Monitor")
    latest, change, movers = get_market_overview()
    c1, c2, c3, c4, c5 = st.columns(5)
    indices_map = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ', '^FTSE': 'FTSE 100', 'GC=F': 'Gold'}
    cols = [c1, c2, c3, c4, c5]
    for i, (ticker, name) in enumerate(indices_map.items()):
        if ticker in latest.index:
            cols[i].metric(name, f"{latest[ticker]:,.2f}", f"{change[ticker]:+.2f}%")
        else: cols[i].metric(name, "N/A", "N/A")
    st.divider()
    st.subheader("Top Daily Movers")
    if not movers.empty:
        c1, c2 = st.columns(2)
        with c1: st.success("🚀 Top Gainers"); st.dataframe(movers.sort_values('Change (%)', ascending=False).head(5))
        with c2: st.error("🔻 Top Losers"); st.dataframe(movers.sort_values('Change (%)', ascending=True).head(5))

# TAB 2: NEWS
with tab2:
    st.header("News & Trading Intelligence")
    sector_map = {
        "Aerospace & Defense": ['BA', 'LMT', 'RTX', 'NOC', 'GD'],
        "Future Tech & Speculative": ['ACHR', 'JOBY', 'RKLB', 'ASTS', 'SPCE'],
        "Technology": ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMD'],
        "Finance (Banking)": ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        "Energy": ['XOM', 'CVX', 'SHEL', 'BP', 'COP'],
        "Healthcare": ['LLY', 'JNJ', 'PFE', 'MRK', 'ABBV']
    }
    col_sel, col_manual = st.columns([1, 2])
    with col_sel: selected_sector = st.selectbox("Select Sector", list(sector_map.keys()))
    if selected_sector == "Custom":
        with col_manual: tickers = st.multiselect("Select Tickers", ['BA', 'LMT', 'RTX', 'NOC', 'ACHR'], default=['BA'])
    else: tickers = sector_map[selected_sector]
    
    if st.button("Run Sector Analysis"):
        st.divider()
        vader = SentimentIntensityAnalyzer()
        url_root = 'https://finviz.com/quote.ashx?t='
        progress_bar = st.progress(0)
        
        # Calculate Logic
        sentiment_summary = []
        all_news_rows = []
        try: tech_data = yf.download(tickers, period="3mo", progress=False)['Close']
        except: tech_data = pd.DataFrame()

        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            # 1. Tech Stats
            rsi_val = 50; buy_zone="N/A"; sell_zone="N/A"; curr=0
            try:
                if not tech_data.empty:
                    ts = tech_data[ticker].dropna() if isinstance(tech_data, pd.DataFrame) else tech_data
                    if not ts.empty:
                        curr = ts.iloc[-1]; rsi_val = calculate_rsi(ts).iloc[-1]
                        u, l = calculate_bollinger_bands_val(ts); buy_zone = f"${l.iloc[-1]:.2f}"; sell_zone = f"${u.iloc[-1]:.2f}"
            except: pass
            
            # 2. News
            avg_score = 0
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = requests.get(url_root + ticker, headers=headers)
                html = BeautifulSoup(req.text, features='html.parser')
                news_table = html.find(id='news-table')
                if news_table:
                    scores = []
                    for row in news_table.findAll('tr')[:5]:
                        if row.a: 
                            s = vader.polarity_scores(row.a.text)['compound']
                            scores.append(s); all_news_rows.append([ticker, row.a.text, s])
                    if scores: avg_score = np.mean(scores)
            except: pass
            
            sentiment_summary.append({"Ticker": ticker, "Price": curr, "RSI": rsi_val, "Sentiment": avg_score})
        
        st.dataframe(pd.DataFrame(sentiment_summary).set_index("Ticker"))
        if all_news_rows: st.dataframe(pd.DataFrame(all_news_rows, columns=['Ticker', 'Headline', 'Score']))

# TAB 3: CHARTING
with tab3:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f'<span class="ticker-text">{sim_ticker}: {name_display}</span>', unsafe_allow_html=True)
        st.markdown(stylized_price_display(curr, prev), unsafe_allow_html=True)
    with c2: 
        if st.button("🤖 Ask AI Analyst"):
            risk, sent, summary = analyze_geopolitics(sim_ticker)
            st.info(f"**AI Report:**\n{summary}")

    full_data = get_stock_data(sim_ticker, "5y", "1d")
    
    if not full_data.empty:
        full_data, pat_keys = detect_comprehensive_patterns(full_data)
        full_data = calculate_indicators(full_data)
        
        if 'tf' not in st.session_state: st.session_state.tf = '1Y'
        
        def slice_df(df, tf):
            if tf == '1D': return df.tail(2)
            if tf == '1W': return df.tail(5)
            if tf == '1M': return df.tail(22)
            if tf == '3M': return df.tail(66)
            if tf == '6M': return df.tail(126)
            if tf == '1Y': return df.tail(252)
            return df
            
        chart_data = slice_df(full_data, st.session_state.tf)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if sub_chart != "None":
            fig, (ax, ax_sub) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax_sub = None

        if chart_style == "Line Chart":
            ax.plot(chart_data['Date'], chart_data['Close'], color='black', lw=1.5)
        else:
            up = chart_data[chart_data['Close'] >= chart_data['Open']]
            down = chart_data[chart_data['Close'] < chart_data['Open']]
            ax.vlines(chart_data['Date'], chart_data['Low'], chart_data['High'], color='gray', lw=1)
            ax.bar(up['Date'], up['Close']-up['Open'], bottom=up['Open'], color='green', width=0.6)
            ax.bar(down['Date'], down['Close']-down['Open'], bottom=down['Open'], color='red', width=0.6)
            
        if "50 SMA" in overlays: ax.plot(chart_data['Date'], chart_data['MA50'], color='blue', ls='--', label='50 SMA')
        if "Bollinger" in overlays: 
            ax.fill_between(chart_data['Date'], chart_data['BB_Upper'], chart_data['BB_Lower'], color='gray', alpha=0.1)
        if "Trend Channel" in overlays:
            tx, ty = calculate_smart_trendline(chart_data)
            if tx is not None: ax.plot(chart_data['Date'], ty[:len(chart_data)], color='blue', alpha=0.5, ls='-.')

        if show_candles:
            bull = chart_data[chart_data.get('Pat_Bullish Engulfing', False) == True]
            if not bull.empty: ax.scatter(bull['Date'], bull['Low']*0.99, marker='^', color='green', s=50, label="Bull Engulf")
            bear = chart_data[chart_data.get('Pat_Bearish Engulfing', False) == True]
            if not bear.empty: ax.scatter(bear['Date'], bear['High']*1.01, marker='v', color='red', s=50, label="Bear Engulf")

        f_dates, paths, actual, low, high, risk, _ = monte_carlo_simulation(sim_ticker, days, sims, vol_scale, backtest, drift_option, use_ai)
        
        if paths is not None:
            ax.fill_between(f_dates, low, high, color='red', alpha=0.1, label='90% Conf.')
            median = np.median(paths, axis=1)
            ax.plot(f_dates, median, color='darkred', ls=':', label='Forecast')
            
            final = paths[-1,:]
            bear = np.percentile(final, 5); bull = np.percentile(final, 95); med = np.median(final)
            buy_z = np.percentile(final, 15); sell_z = np.percentile(final, 85)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Bear Case", f"${bear:.2f}")
            m2.metric("Median", f"${med:.2f}")
            m3.metric("Bull Case", f"${bull:.2f}")
            st.success(f"**Buy Zone:** Below ${buy_z:.2f} | **Sell Zone:** Above ${sell_z:.2f}")
            
            if backtest and actual is not None:
                ax.plot(f_dates, actual, color='blue', lw=2, label='Actual')

        ax.legend(loc='upper left')
        
        if ax_sub is not None:
            if sub_chart == "RSI":
                ax_sub.plot(chart_data['Date'], chart_data['RSI'], color='purple')
                ax_sub.axhline(70, color='red', ls=':'); ax_sub.axhline(30, color='green', ls=':')
            elif sub_chart == "Volume":
                ax_sub.bar(chart_data['Date'], chart_data['Volume'], color='gray') 
            elif sub_chart == "MACD":
                ax_sub.plot(chart_data['Date'], chart_data['MACD_Line'], color='blue')
                ax_sub.plot(chart_data['Date'], chart_data['MACD_Signal'], color='orange')
        
        st.pyplot(fig)
        
        # Timeframe Buttons
        tfs = ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y']
        cols = st.columns(len(tfs))
        for i, tf in enumerate(tfs):
            sub_df = slice_df(full_data, tf)
            if not sub_df.empty:
                s = sub_df['Close'].iloc[0]; e = sub_df['Close'].iloc[-1]
                chg = ((e - s)/s)*100
                label = f"{tf} {chg:+.1f}%"
            else: label = tf
            if cols[i].button(label): st.session_state.tf = tf; st.rerun()

        st.subheader("Candlestick Reliability")
        acc_table = calculate_pattern_accuracy(full_data, pat_keys)
        if not acc_table.empty: st.dataframe(acc_table, use_container_width=True)

# TAB 4: SECTOR SCANNER
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
    sec = st.selectbox("Select Sector", list(sector_dict.keys()))
    filter_option = st.radio("Filter By:", ["All", "Upcoming Earnings (30d)", "High Dividend (>3%)"], horizontal=True)
    
    if st.button("🚀 Run Sector Simulation"):
        ts = sector_dict[sec]
        res = []
        bar = st.progress(0)
        filtered_ts = []
        if filter_option == "All": filtered_ts = ts
        else:
            for t in ts:
                try:
                    tick = yf.Ticker(t)
                    if filter_option == "High Dividend (>3%)":
                        div = tick.info.get('dividendYield', 0)
                        if div and div > 0.03: filtered_ts.append(t)
                    elif filter_option == "Upcoming Earnings (30d)": filtered_ts.append(t)
                except: pass
            if not filtered_ts: st.warning("No stocks matched filter. Showing all."); filtered_ts = ts

        for i, t in enumerate(filtered_ts):
            bar.progress((i+1)/len(filtered_ts))
            # FIX: Unpack 7 values
            _, paths, _, _, _, _, _ = monte_carlo_simulation(t, 30, 100, 0.5, False)
            if paths is not None:
                curr = paths[0,0]; pred = np.median(paths[-1,:])
                ret = ((pred - curr)/curr)*100
                res.append({"Ticker": t, "Current": curr, "Forecast (30d)": pred, "Return": ret})
        
        if res:
            st.dataframe(pd.DataFrame(res).style.format({"Current": "${:.2f}", "Forecast (30d)": "${:.2f}", "Return": "{:+.2f}%"}))

# TAB 5: WHALES
with tab5:
    st.header("Whale & Competitor Tracker")
    st.subheader("🏛️ Congressional Trading (Live)")
    components.iframe("https://www.capitoltrades.com/trades", height=600, scrolling=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("🏰 Warren Buffett")
        st.dataframe(get_buffett_portfolio())
    with c2:
        st.subheader("🕵️ Corporate Insiders")
        st.dataframe(get_insider_trading())

# TAB 6: PORTFOLIO
with tab6:
    st.header("Portfolio Analytics")
    sel = st.multiselect("Assets", ['BA', 'LMT', 'RTX', 'NOC', '^GSPC'], default=['BA', '^GSPC'])
    if st.button("Run Analysis"):
        df = yf.download(sel, period="1y")['Close'].ffill().dropna()
        st.line_chart((df/df.iloc[0]-1)*100)
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(); sns.heatmap(df.pct_change().corr(), annot=True, ax=ax); st.pyplot(fig)

# TAB 7: VALUATION
with tab7:
    st.header("💰 Valuation & Risk")
    st.subheader("1. Interactive DCF")
    if fund_data:
        fcf = fund_data.get('Free Cash Flow'); shares = fund_data.get('Shares Outstanding')
        if fcf and shares:
            per_share = fcf/shares
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"FCF/Share: ${per_share:.2f}")
                g = st.slider("Growth %", 0, 50, 10)/100
                d = st.slider("Discount %", 5, 20, 9)/100
            with c2:
                fut = [per_share * ((1+g)**i) for i in range(1,6)]
                term = fut[-1] * 15
                dcf = sum([f/((1+d)**(i+1)) for i,f in enumerate(fut)]) + (term/((1+d)**5))
                st.metric("Fair Value", f"${dcf:.2f}")
    st.divider()
    st.subheader("2. Value at Risk (VaR)")
    try:
        hist = yf.download(sim_ticker, period="1y")['Close'].pct_change().dropna()
        var = np.percentile(hist, 5)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(hist, kde=True, ax=ax)
        ax.axvline(var, color='red', ls='--')
        st.pyplot(fig)
        st.error(f"95% VaR: {var:.2%}")
    except: st.error("Data error")
