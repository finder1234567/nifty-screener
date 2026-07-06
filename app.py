import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ssl
import io
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# MAC FIX: Bypasses the SSL certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set the page layout
st.set_page_config(page_title="Indian Market Screener", layout="wide")
st.title("🔥 Ultimate Nifty 50 Screener: Absolute Verdict Edition")
st.write("Scanning NSE equities for technical signals, momentum, news sentiment, and generating Absolute Confluence verdicts.")

nifty_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TMPV.NS",
]

TICKER_TO_SEARCH_NAME = {
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals", "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics", "BHARTIARTL.NS": "Bharti Airtel",
    "CIPLA.NS": "Cipla", "COALINDIA.NS": "Coal India",
    "DRREDDY.NS": "Dr Reddys Laboratories", "EICHERMOT.NS": "Eicher Motors",
    "ETERNAL.NS": "Eternal Zomato", "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCLTech", "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life", "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "INDIGO.NS": "IndiGo InterGlobe Aviation", "INFY.NS": "Infosys",
    "ITC.NS": "ITC Limited", "JIOFIN.NS": "Jio Financial Services",
    "JSWSTEEL.NS": "JSW Steel", "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen Toubro", "M&M.NS": "Mahindra Mahindra",
    "MARUTI.NS": "Maruti Suzuki", "MAXHEALTH.NS": "Max Healthcare",
    "NESTLEIND.NS": "Nestle India", "NTPC.NS": "NTPC Limited",
    "ONGC.NS": "Oil Natural Gas Corporation", "POWERGRID.NS": "Power Grid Corporation",
    "RELIANCE.NS": "Reliance Industries", "SBILIFE.NS": "SBI Life Insurance",
    "SHRIRAMFIN.NS": "Shriram Finance", "SBIN.NS": "State Bank of India",
    "SUNPHARMA.NS": "Sun Pharmaceutical", "TCS.NS": "Tata Consultancy Services",
    "TATACONSUM.NS": "Tata Consumer Products", "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra", "TITAN.NS": "Titan Company",
    "TRENT.NS": "Trent Limited", "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro Limited", "TMPV.NS": "Tata Motors Passenger Vehicles",
}

PREFERRED_NEWS_SOURCES = [
    ("moneycontrol.com", "Moneycontrol"),
    ("screener.in", "Screener"),
    ("groww.in", "Groww"),
]

def _fetch_google_news_rss(encoded_query, max_items, timeout):
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw_data = response.read()

    root = ET.fromstring(raw_data)
    items = root.findall('.//item')[:max_items]
    parsed = []
    for item in items:
        title = item.findtext('title', default='')
        pub_date = item.findtext('pubDate', default='')
        if title:
            parsed.append({'title': title, 'pubDate': pub_date})
    return parsed

def fetch_stock_news(query, max_items_per_source=3, timeout=8):
    news_list = []
    for domain, label in PREFERRED_NEWS_SOURCES:
        try:
            encoded_query = urllib.parse.quote(f"{query} site:{domain}")
            items = _fetch_google_news_rss(encoded_query, max_items_per_source, timeout)
            for item in items: item['source'] = label
            news_list.extend(items)
        except Exception:
            continue
    if not news_list:
        try:
            encoded_query = urllib.parse.quote(f"{query} stock NSE")
            items = _fetch_google_news_rss(encoded_query, 5, timeout)
            for item in items: item['source'] = "Google News"
            news_list.extend(items)
        except Exception:
            pass
    return news_list

# =========================================================
# --- INDICATOR MATH ---
# =========================================================

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / np.maximum(loss, 1e-8)
    return 100 - (100 / (1 + rs))

def calculate_roc(data, window=10):
    return ((data - data.shift(window)) / data.shift(window)) * 100

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * (df['Close'] - low_min) / np.maximum((high_max - low_min), 1e-8)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_williams_r(df, period=14):
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    wr = -100 * (high_max - df['Close']) / np.maximum((high_max - low_min), 1e-8)
    return wr

def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * np.maximum(mean_dev, 1e-8))
    return cci

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(df, period=14):
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = calculate_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / np.maximum(atr, 1e-8)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / np.maximum(atr, 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / np.maximum((plus_di + minus_di), 1e-8)
    return dx.rolling(window=period).mean()

def calculate_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def calculate_vwap(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vol_sum = df['Volume'].rolling(window=window).sum()
    return (typical_price * df['Volume']).rolling(window=window).sum() / np.maximum(vol_sum, 1e-8)

def calculate_supertrend(df, period=10, multiplier=3):
    atr = calculate_atr(df, period).values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    hl2 = (high + low) / 2

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = np.zeros(len(df))
    direction = np.empty(len(df), dtype=object)

    for i in range(len(df)):
        if i == 0:
            supertrend[i] = upperband[i]
            direction[i] = "Sell"
            continue

        if close[i] > upperband[i - 1]:
            direction[i] = "Buy"
        elif close[i] < lowerband[i - 1]:
            direction[i] = "Sell"
        else:
            direction[i] = direction[i - 1]
            if direction[i] == "Buy" and lowerband[i] < lowerband[i - 1]:
                lowerband[i] = lowerband[i - 1]
            if direction[i] == "Sell" and upperband[i] > upperband[i - 1]:
                upperband[i] = upperband[i - 1]

        supertrend[i] = lowerband[i] if direction[i] == "Buy" else upperband[i]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

# =========================================================
# --- NEWS SENTIMENT ANALYZER ---
# =========================================================

POSITIVE_WORDS = ['surge', 'profit', 'jump', 'buy', 'growth', 'up', 'dividend', 'beat', 'high',
                   'win', 'upgrade', 'bull', 'rally', 'record', 'outperform', 'soar', 'gain',
                   'strong', 'expansion', 'boost']
NEGATIVE_WORDS = ['fall', 'loss', 'drop', 'sell', 'decline', 'down', 'miss', 'low', 'penalty',
                   'downgrade', 'crash', 'bear', 'plunge', 'weak', 'probe', 'fraud', 'lawsuit',
                   'cut', 'slump', 'default']

def analyze_sentiment(news_list):
    if not news_list:
        return {"label": "No News ⚪", "score": 0, "headline": "-", "source": "-", "count": 0}

    scored_headlines = []
    for item in news_list[:9]:
        title = item.get('title', '')
        if not title: continue
        source = item.get('source', 'Google News')
        text = title.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)
        scored_headlines.append((title, pos - neg, pos, neg, source))

    if not scored_headlines:
        return {"label": "No News ⚪", "score": 0, "headline": "-", "source": "-", "count": 0}

    total_pos = sum(h[2] for h in scored_headlines)
    total_neg = sum(h[3] for h in scored_headlines)
    net_score = total_pos - total_neg

    trigger = max(scored_headlines, key=lambda h: abs(h[1]))
    trigger_headline = trigger[0]
    trigger_source = trigger[4]

    if net_score > 0: label = "Positive 🟢"
    elif net_score < 0: label = "Negative 🔴"
    else: label = "Neutral ⚪"

    return {
        "label": label,
        "score": net_score,
        "headline": trigger_headline,
        "source": trigger_source,
        "count": len(scored_headlines),
    }

# =========================================================
# --- MAIN SCANNER ---
# =========================================================

def process_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)

        try:
            info = stock.info
        except Exception:
            info = {}

        pe_ratio = info.get('trailingPE', info.get('forwardPE', "N/A"))
        if isinstance(pe_ratio, (int, float)):
            pe_ratio = round(pe_ratio, 2)

        mkt_cap = info.get('marketCap', "N/A")
        mkt_cap_str = f"₹{mkt_cap / 10000000:,.0f} Cr" if isinstance(mkt_cap, (int, float)) else "N/A"

        df = stock.history(period="1y")
        if len(df) == 0:
            return None

        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['ROC10'] = calculate_roc(df['Close'], window=10)
        df['StochK'], df['StochD'] = calculate_stochastic(df)
        df['WilliamsR'] = calculate_williams_r(df)
        df['CCI'] = calculate_cci(df)

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['ATR'] = calculate_atr(df)
        df['Vol_Avg_20'] = df['Volume'].rolling(window=20).mean()
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['ADX'] = calculate_adx(df)
        df['OBV'] = calculate_obv(df)
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
        df['VWAP20'] = calculate_vwap(df, window=20)
        df['Supertrend'], df['Supertrend_Dir'] = calculate_supertrend(df)

        last_close = float(df['Close'].iloc[-1])
        bb_upper = float(df['BB_Upper'].iloc[-1])
        bb_lower = float(df['BB_Lower'].iloc[-1])
        high_52w = float(df['High'].max())
        low_52w = float(df['Low'].min())
        pct_from_high = ((last_close - high_52w) / high_52w) * 100
        pct_from_low = ((last_close - low_52w) / low_52w) * 100

        # Earnings Date Logic
        earnings_date = "N/A"
        days_to_earnings = None
        earnings_flag = "-"
        try:
            calendar = stock.calendar
            if calendar:
                dates = calendar.get('Earnings Date', []) if isinstance(calendar, dict) else (calendar['Earnings Date'] if 'Earnings Date' in calendar else [])
                if len(dates) > 0:
                    ed = pd.to_datetime(dates[0])
                    if ed.tzinfo is not None:
                        ed = ed.tz_localize(None)
                    earnings_date = ed.strftime('%Y-%m-%d')
                    days_to_earnings = (ed.normalize() - pd.Timestamp.now().normalize()).days
                    if days_to_earnings is not None:
                        if 0 <= days_to_earnings <= 7: earnings_flag = f"⚠️ In {days_to_earnings}d"
                        elif 8 <= days_to_earnings <= 21: earnings_flag = f"Upcoming in {days_to_earnings}d"
                        elif days_to_earnings < 0: earnings_flag = "Recently Reported"
                        else: earnings_flag = f"In {days_to_earnings}d"
        except Exception:
            pass

        try:
            search_name = TICKER_TO_SEARCH_NAME.get(ticker, ticker.replace(".NS", ""))
            news_items = fetch_stock_news(search_name)
            sentiment_info = analyze_sentiment(news_items)
        except Exception:
            sentiment_info = {"label": "Fetch Error", "score": 0, "headline": "-", "source": "-", "count": 0}

        trend = "Bullish 🟢" if float(df['MA50'].iloc[-1]) > float(df['MA200'].iloc[-1]) else "Bearish 🔴"
        macd_signal = "Bull 🟢" if float(df['MACD'].iloc[-1]) > float(df['Signal_Line'].iloc[-1]) else "Bear 🔴"
        vol_signal = "High 🚀" if float(df['Volume'].iloc[-1]) > (float(df['Vol_Avg_20'].iloc[-1]) * 1.5) else "Normal"

        latest_rsi = float(df['RSI'].iloc[-1])
        rsi_signal = "Oversold" if latest_rsi < 30 else ("Overbought" if latest_rsi > 70 else "Neutral")
        bb_signal = "Breaking Upper Band 🔥" if last_close > bb_upper else ("Below Lower Band 🧊" if last_close < bb_lower else "Inside Bands")
        latest_roc = float(df['ROC10'].iloc[-1])
        roc_signal = "Accelerating 🚀" if latest_roc > 3 else ("Decelerating 🐢" if latest_roc < -3 else "Flat")
        
        latest_stoch_k = float(df['StochK'].iloc[-1])
        stoch_signal = "Overbought" if latest_stoch_k > 80 else ("Oversold" if latest_stoch_k < 20 else "Neutral")
        
        latest_wr = float(df['WilliamsR'].iloc[-1])
        wr_signal = "Overbought" if latest_wr > -20 else ("Oversold" if latest_wr < -80 else "Neutral")
        
        latest_cci = float(df['CCI'].iloc[-1])
        cci_signal = "Overbought" if latest_cci > 100 else ("Oversold" if latest_cci < -100 else "Neutral")
        
        latest_adx = float(df['ADX'].iloc[-1])
        adx_signal = "Strong Trend 💪" if latest_adx > 25 else "Weak/No Trend"
        obv_signal = "Rising 📈" if float(df['OBV'].iloc[-1]) > float(df['OBV_MA20'].iloc[-1]) else "Falling 📉"
        latest_vwap = float(df['VWAP20'].iloc[-1])
        vwap_signal = "Above VWAP 🟢" if last_close > latest_vwap else "Below VWAP 🔴"
        supertrend_signal = "Buy 🟢" if df['Supertrend_Dir'].iloc[-1] == "Buy" else "Sell 🔴"

        # Math Scoring Engine
        score = 0
        score += 1 if trend.startswith("Bullish") else -1
        score += 1 if macd_signal.startswith("Bull") else -1
        score += 1 if latest_roc > 0 else -1
        score += 1 if latest_stoch_k > 50 else -1
        score += 1 if latest_wr > -50 else -1
        score += 1 if latest_cci > 0 else -1
        score += 1 if obv_signal.startswith("Rising") else -1
        score += 1 if vwap_signal.startswith("Above") else -1
        score += 1 if supertrend_signal.startswith("Buy") else -1
        
        if latest_adx > 25: score = score * 1.2
        sentiment_nudge = max(-1, min(1, sentiment_info["score"]))
        score += sentiment_nudge

        momentum_score = round(score, 1)
        if momentum_score >= 5: momentum_label = "Strong Bullish 🚀"
        elif momentum_score >= 1: momentum_label = "Mild Bullish 🟢"
        elif momentum_score <= -5: momentum_label = "Strong Bearish 🔻"
        elif momentum_score <= -1: momentum_label = "Mild Bearish 🔴"
        else: momentum_label = "Neutral ⚪"

        # Base Recommendation
        if momentum_score >= 3: recommendation = "BUY 🟢"
        elif momentum_score <= -3: recommendation = "SELL 🔴"
        else: recommendation = "HOLD ⚪"

        overbought_count = sum([rsi_signal == "Overbought", stoch_signal == "Overbought", cci_signal == "Overbought"])
        oversold_count = sum([rsi_signal == "Oversold", stoch_signal == "Oversold", cci_signal == "Oversold"])

        if recommendation == "BUY 🟢" and overbought_count >= 2:
            recommendation = "BUY ⚠️ (Overbought)"
        elif recommendation == "SELL 🔴" and oversold_count >= 2:
            recommendation = "SELL ⚠️ (Oversold)"

        # --- NEW: ABSOLUTE CONVICTION VERDICT ---
        earnings_risk = "⚠️" in earnings_flag
        news_score = sentiment_info['score']
        high_vol = vol_signal.startswith("High")

        if earnings_risk:
            absolute_verdict = "HOLD ⚠️ (Earnings Risk)"
        elif momentum_score >= 4 and news_score > 0 and high_vol:
            absolute_verdict = "STRONG BUY 🌟"
        elif momentum_score <= -4 and news_score < 0 and high_vol:
            absolute_verdict = "STRONG SELL 💥"
        elif momentum_score >= 3 and news_score < 0:
            absolute_verdict = "HOLD ⚖️ (Bad News)"
        elif momentum_score <= -3 and news_score > 0:
            absolute_verdict = "HOLD ⚖️ (Good News)"
        elif recommendation.startswith("BUY"):
            absolute_verdict = "BUY 🟢"
        elif recommendation.startswith("SELL"):
            absolute_verdict = "SELL 🔴"
        else:
            absolute_verdict = "HOLD ⚪"

        return {
            "Stock": ticker.replace(".NS", ""),
            "Absolute Verdict": absolute_verdict,
            "Recommendation": recommendation,
            "Momentum": momentum_label,
            "Momentum Score": momentum_score,
            "Trend (MA)": trend,
            "Price (₹)": round(last_close, 2),
            "Market Cap": mkt_cap_str,
            "P/E Ratio": pe_ratio,
            "News Sentiment": f"{sentiment_info['label']} ({sentiment_info['score']:+d})" if sentiment_info['score'] != 0 or sentiment_info['label'] != 'No News ⚪' else sentiment_info['label'],
            "Sentiment Trigger Headline": sentiment_info["headline"],
            "News Source": sentiment_info["source"],
            "Next Earnings": earnings_date,
            "Earnings Watch": earnings_flag,
            "MACD": macd_signal,
            "RSI": rsi_signal,
            "ROC(10d)": f"{latest_roc:.1f}% ({roc_signal})",
            "Stochastic": f"{latest_stoch_k:.0f} ({stoch_signal})",
            "Williams %R": f"{latest_wr:.0f} ({wr_signal})",
            "CCI": f"{latest_cci:.0f} ({cci_signal})",
            "ADX": f"{latest_adx:.0f} ({adx_signal})",
            "OBV": obv_signal,
            "VWAP(20d)": vwap_signal,
            "Supertrend": supertrend_signal,
            "Bollinger": bb_signal,
            "ATR": round(float(df['ATR'].iloc[-1]), 2),
            "52W High %": f"{pct_from_high:.1f}%",
            "52W Low %": f"{pct_from_low:.1f}%",
            "Volume": vol_signal,
        }
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_and_analyze(tickers):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers]
        for future in as_completed(futures):
            row = future.result()
            if row: results.append(row)
    return pd.DataFrame(results)

# =========================================================
# --- BACKTESTING ENGINE ---
# =========================================================

BACKTEST_HORIZONS = (5, 10, 20)

def compute_vectorized_signals(df):
    d = df.copy()
    d['MA50'] = d['Close'].rolling(window=50).mean()
    d['MA200'] = d['Close'].rolling(window=200).mean()
    d['RSI'] = calculate_rsi(d['Close'])
    d['ROC10'] = calculate_roc(d['Close'], window=10)
    d['StochK'], d['StochD'] = calculate_stochastic(d)
    d['WilliamsR'] = calculate_williams_r(d)
    d['CCI'] = calculate_cci(d)

    exp1 = d['Close'].ewm(span=12, adjust=False).mean()
    exp2 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = exp1 - exp2
    d['Signal_Line'] = d['MACD'].ewm(span=9, adjust=False).mean()

    d['ATR'] = calculate_atr(d)
    d['ADX'] = calculate_adx(d)
    d['OBV'] = calculate_obv(d)
    d['OBV_MA20'] = d['OBV'].rolling(window=20).mean()
    d['VWAP20'] = calculate_vwap(d, window=20)
    d['Supertrend'], d['Supertrend_Dir'] = calculate_supertrend(d)

    warmup_cols = ['MA200', 'RSI', 'StochK', 'WilliamsR', 'CCI', 'MACD', 'ADX', 'OBV_MA20', 'VWAP20']
    valid_mask = d[warmup_cols].notna().all(axis=1)

    trend_bull = d['MA50'] > d['MA200']
    macd_bull = d['MACD'] > d['Signal_Line']
    roc_pos = d['ROC10'] > 0
    stoch_bull = d['StochK'] > 50
    wr_bull = d['WilliamsR'] > -50
    cci_bull = d['CCI'] > 0
    obv_rising = d['OBV'] > d['OBV_MA20']
    vwap_above = d['Close'] > d['VWAP20']
    supertrend_buy = d['Supertrend_Dir'] == 'Buy'

    score = (
        (trend_bull.astype(int) * 2 - 1) + (macd_bull.astype(int) * 2 - 1) +
        (roc_pos.astype(int) * 2 - 1) + (stoch_bull.astype(int) * 2 - 1) +
        (wr_bull.astype(int) * 2 - 1) + (cci_bull.astype(int) * 2 - 1) +
        (obv_rising.astype(int) * 2 - 1) + (vwap_above.astype(int) * 2 - 1) +
        (supertrend_buy.astype(int) * 2 - 1)
    )
    strong_trend = d['ADX'] > 25
    score = np.where(strong_trend, score * 1.2, score)
    d['Momentum_Score_Hist'] = np.where(valid_mask, np.round(score, 1), np.nan)

    conditions = [
        d['Momentum_Score_Hist'] >= 5, d['Momentum_Score_Hist'] >= 1,
        d['Momentum_Score_Hist'] <= -5, d['Momentum_Score_Hist'] <= -1,
    ]
    choices = ['Strong Bullish 🚀', 'Mild Bullish 🟢', 'Strong Bearish 🔻', 'Mild Bearish 🔴']
    d['Momentum_Label_Hist'] = pd.Series(np.select(conditions, choices, default='Neutral ⚪'), index=d.index).astype(object)
    d.loc[~valid_mask, 'Momentum_Label_Hist'] = np.nan

    rsi_overbought, rsi_oversold = d['RSI'] > 70, d['RSI'] < 30
    stoch_overbought, stoch_oversold = d['StochK'] > 80, d['StochK'] < 20
    cci_overbought, cci_oversold = d['CCI'] > 100, d['CCI'] < -100
    overbought_count = rsi_overbought.astype(int) + stoch_overbought.astype(int) + cci_overbought.astype(int)
    oversold_count = rsi_oversold.astype(int) + stoch_oversold.astype(int) + cci_oversold.astype(int)

    base_reco = np.where(d['Momentum_Score_Hist'] >= 3, 'BUY', np.where(d['Momentum_Score_Hist'] <= -3, 'SELL', 'HOLD'))
    reco = np.where((base_reco == 'BUY') & (overbought_count >= 2), 'BUY ⚠️ (Overbought)',
            np.where((base_reco == 'SELL') & (oversold_count >= 2), 'SELL ⚠️ (Oversold)',
             np.where(base_reco == 'BUY', 'BUY 🟢', np.where(base_reco == 'SELL', 'SELL 🔴', 'HOLD ⚪'))))
    
    d['Recommendation_Hist'] = pd.Series(reco, index=d.index).astype(object)
    d.loc[~valid_mask, 'Recommendation_Hist'] = np.nan

    return d

def add_forward_returns(d, horizons=BACKTEST_HORIZONS):
    for h in horizons: d[f'Fwd_Return_{h}d'] = d['Close'].shift(-h) / d['Close'] - 1
    return d

def backtest_ticker(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{years}y")
        if len(df) < 250: return None
        d = compute_vectorized_signals(df)
        d = add_forward_returns(d)
        d = d.dropna(subset=['Momentum_Score_Hist'])
        if d.empty: return None
        d['Stock'] = ticker.replace(".NS", "")
        cols = ['Stock', 'Momentum_Label_Hist', 'Recommendation_Hist'] + [f'Fwd_Return_{h}d' for h in BACKTEST_HORIZONS]
        return d[cols]
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def run_backtest_all(tickers, years):
    frames = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(backtest_ticker, t, years) for t in tickers]
        for future in as_completed(futures):
            res = future.result()
            if res is not None and not res.empty: frames.append(res)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def summarize_backtest(combined, group_col, horizons=BACKTEST_HORIZONS):
    rows = []
    for label, group in combined.groupby(group_col):
        row = {group_col: label, 'Occurrences': len(group)}
        for h in horizons:
            valid = group[f'Fwd_Return_{h}d'].dropna()
            if len(valid) > 0:
                row[f'Win Rate {h}d'] = f"{(valid > 0).mean() * 100:.0f}%"
                row[f'Avg Return {h}d'] = f"{valid.mean() * 100:+.2f}%"
            else:
                row[f'Win Rate {h}d'] = "-"
                row[f'Avg Return {h}d'] = "-"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    baseline_row = {group_col: 'ALL DAYS (Baseline)', 'Occurrences': len(combined)}
    for h in horizons:
        valid = combined[f'Fwd_Return_{h}d'].dropna()
        baseline_row[f'Win Rate {h}d'] = f"{(valid > 0).mean() * 100:.0f}%"
        baseline_row[f'Avg Return {h}d'] = f"{valid.mean() * 100:+.2f}%"

    summary_df = pd.concat([summary_df, pd.DataFrame([baseline_row])], ignore_index=True)
    return summary_df

if st.button("🚀 Run Ultimate Scan"):
    with st.spinner('Downloading charts, fundamentals, momentum, and news data...'):
        analysis_df = fetch_and_analyze(tuple(nifty_tickers))

        if analysis_df.empty:
            st.error("Failed to download data.")
            st.session_state.pop("analysis_df", None)
        else:
            analysis_df = analysis_df.sort_values(by="Momentum Score", ascending=False).reset_index(drop=True)

            if "backtest_combined" in st.session_state:
                reco_summary = summarize_backtest(st.session_state["backtest_combined"], "Recommendation_Hist")
                win_map = dict(zip(reco_summary["Recommendation_Hist"], reco_summary["Win Rate 10d"]))
                ret_map = dict(zip(reco_summary["Recommendation_Hist"], reco_summary["Avg Return 10d"]))
                occ_map = dict(zip(reco_summary["Recommendation_Hist"], reco_summary["Occurrences"]))
                analysis_df["Hist. Win Rate (10d)"] = analysis_df["Recommendation"].map(win_map).fillna("-")
                analysis_df["Hist. Avg Return (10d)"] = analysis_df["Recommendation"].map(ret_map).fillna("-")
                analysis_df["Hist. Occurrences"] = analysis_df["Recommendation"].map(occ_map).fillna(0).astype(int)

                cols = list(analysis_df.columns)
                for c in ["Hist. Win Rate (10d)", "Hist. Avg Return (10d)", "Hist. Occurrences"]:
                    cols.remove(c)
                # Insert the historical data directly after the new Absolute Verdict column
                insert_at = cols.index("Absolute Verdict") + 1
                for i, c in enumerate(["Hist. Win Rate (10d)", "Hist. Avg Return (10d)", "Hist. Occurrences"]):
                    cols.insert(insert_at + i, c)
                analysis_df = analysis_df[cols]

            st.session_state["analysis_df"] = analysis_df
            st.session_state["scan_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")

if "analysis_df" in st.session_state:
    st.success("Scan Complete!")
    st.dataframe(st.session_state["analysis_df"], width="stretch")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state["analysis_df"].to_excel(writer, index=False, sheet_name="Nifty50_Screener")
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download as Excel",
        data=excel_buffer,
        file_name=f"nifty50_screener_{st.session_state['scan_time']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.subheader("📊 Historical Signal Backtest")
st.caption(
    "Tests whether the Momentum/Recommendation signals above have actually preceded "
    "positive returns in the past, pooled across all 50 stocks. Excludes news sentiment "
    "(no historical headline archive available) - this validates the technical signal "
    "logic only, not any individual stock's fundamentals or news."
)

backtest_years = st.slider("Years of history to test", min_value=2, max_value=10, value=5)

if st.button("🔬 Run Backtest"):
    with st.spinner(f"Backtesting {len(nifty_tickers)} stocks over {backtest_years} years of history..."):
        combined = run_backtest_all(tuple(nifty_tickers), backtest_years)
        if combined.empty:
            st.error("Backtest failed - no historical data retrieved.")
        else:
            st.session_state["backtest_combined"] = combined
            st.session_state["backtest_years"] = backtest_years

if "backtest_combined" in st.session_state:
    combined = st.session_state["backtest_combined"]
    st.success(
        f"Backtested {combined['Stock'].nunique()} stocks across {len(combined):,} "
        f"trading days over {st.session_state['backtest_years']} years."
    )

    st.write("**By Recommendation Signal**")
    reco_summary = summarize_backtest(combined, "Recommendation_Hist")
    st.dataframe(reco_summary, width="stretch")

    st.write("**By Momentum Label**")
    momentum_summary = summarize_backtest(combined, "Momentum_Label_Hist")
    st.dataframe(momentum_summary, width="stretch")