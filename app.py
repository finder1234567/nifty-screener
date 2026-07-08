import os
import sys
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from pykrx import stock as krx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# =========================================================================
# STATIC FALLBACK UNIVERSE
# =========================================================================
FALLBACK_KOSPI200 = {
    "005930.KS": "Samsung Electronics", "000660.KS": "SK Hynix", "373220.KS": "LG Energy Solution",
    "207940.KS": "Samsung Biologics", "005380.KS": "Hyundai Motor", "000270.KS": "Kia",
    "068270.KS": "Celltrion", "005490.KS": "POSCO Holdings", "035420.KS": "NAVER",
    "006400.KS": "Samsung SDI", "051910.KS": "LG Chem", "105560.KS": "KB Financial Group",
    "055550.KS": "Shinhan Financial Group", "086790.KS": "Hana Financial Group", "012330.KS": "Hyundai Mobis",
    "096770.KS": "SK Innovation", "066570.KS": "LG Electronics", "035720.KS": "Kakao",
    "032830.KS": "Samsung Life Insurance", "329180.KS": "HD Hyundai Heavy Industries",
    "015760.KS": "Korea Electric Power", "017670.KS": "SK Telecom", "030200.KS": "KT Corp",
    "009830.KS": "Hanwha Solutions", "010950.KS": "S-Oil", "011170.KS": "Lotte Chemical",
    "000810.KS": "Samsung Fire & Marine Insurance", "090430.KS": "Amorepacific", "032640.KS": "LG Uplus",
    "034020.KS": "Doosan Enerbility", "011200.KS": "HMM", "047810.KS": "Korea Aerospace Industries",
    "326030.KS": "SK Biopharmaceuticals", "138040.KS": "Meritz Financial Group", "097950.KS": "CJ CheilJedang",
    "316140.KS": "Woori Financial Group", "028050.KS": "Samsung Engineering", "012450.KS": "Hanwha Aerospace"
}

class DummyOutput:
    def write(self, x): pass
    def flush(self): pass


# =========================================================================
# BUILT-IN TECHNICAL INDICATOR CALCULATIONS
# =========================================================================
def calculate_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd_hist.fillna(0)

def calculate_atr(high, low, close, window=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.fillna(0)

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def calculate_stochastic(high, low, close, window=14, smooth=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k.rolling(window=smooth).mean()
    return k.fillna(50)

def calculate_cci(high, low, close, window=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci.fillna(0)

def calculate_adx(high, low, close, window=14):
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(high, low, close, window=1)
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum
    adx = dx.rolling(window=window).mean()
    return adx.fillna(20)

def calculate_obv(close, volume):
    """On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_vwap(high, low, close, volume, window=20):
    """Volume Weighted Average Price"""
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap.fillna(close)

def calculate_williams_r(high, low, close, window=14):
    """Williams %R"""
    highest = high.rolling(window=window).max()
    lowest = low.rolling(window=window).min()
    wr = -100 * (highest - close) / (highest - lowest)
    return wr.fillna(-50)

# =========================================================================
# TICKER UNIVERSE FETCH 
# =========================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_kospi200_tickers():
    if not PYKRX_AVAILABLE:
        return dict(FALLBACK_KOSPI200)
        
    old_stdout = sys.stdout
    sys.stdout = DummyOutput()
    
    for i in range(1, 10):
        safe_date = (datetime.today() - timedelta(days=i)).strftime("%Y%m%d")
        try:
            tickers = krx_stock.get_index_portfolio_deposit_file(safe_date, "1028")
            if not tickers:
                tickers = krx_stock.get_index_portfolio_deposit_file("1028", safe_date)
            if tickers:
                kospi_dict = {}
                for t in tickers:
                    try: 
                        kospi_dict[f"{t}.KS"] = krx_stock.get_market_ticker_name(t)
                    except: 
                        kospi_dict[f"{t}.KS"] = f"Unknown ({t})"
                if len(kospi_dict) > 50:
                    sys.stdout = old_stdout
                    return kospi_dict
        except: 
            pass
            
    for i in range(1, 5):
        safe_date = (datetime.today() - timedelta(days=i)).strftime("%Y%m%d")
        try:
            tickers = krx_stock.get_market_ticker_list(safe_date, market="KOSPI")
            if tickers:
                kospi_dict = {}
                for t in tickers[:100]:
                    try: 
                        kospi_dict[f"{t}.KS"] = krx_stock.get_market_ticker_name(t)
                    except: 
                        kospi_dict[f"{t}.KS"] = f"KOSPI Stock ({t})"
                if len(kospi_dict) > 50:
                    sys.stdout = old_stdout
                    return kospi_dict
        except: 
            pass
            
    sys.stdout = old_stdout
    return dict(FALLBACK_KOSPI200)


# =========================================================================
# PE RATIO CALCULATION ENGINE
# =========================================================================
def calculate_pe_ratio(ticker: str, info: dict, df: pd.DataFrame) -> str:
    """Calculate P/E ratio with multiple fallback methods"""
    
    trailing_pe = info.get("trailingPE")
    if trailing_pe is not None and trailing_pe > 0:
        try:
            return f"{float(trailing_pe):.2f}"
        except (ValueError, TypeError):
            pass
    
    forward_pe = info.get("forwardPE")
    if forward_pe is not None and forward_pe > 0:
        try:
            return f"{float(forward_pe):.2f}"
        except (ValueError, TypeError):
            pass
    
    try:
        price = info.get("currentPrice") or df["Close"].iloc[-1]
        eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
        
        if price and eps and float(eps) != 0:
            pe = float(price) / float(eps)
            if pe > 0 and pe < 1000:
                return f"{pe:.2f}"
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    
    try:
        dividend_yield = info.get("dividendYield")
        if dividend_yield and float(dividend_yield) > 0:
            payout_ratio = info.get("payoutRatio", 0.5)
            if payout_ratio > 0:
                implied_pe = dividend_yield / payout_ratio
                if 0 < implied_pe < 1000:
                    return f"{implied_pe:.2f}"
    except (ValueError, TypeError):
        pass
    
    return "N/A"


# =========================================================================
# ADVANCED INDICATORS MATHEMATICAL ENGINE
# =========================================================================
def calculate_supertrend(df, period=7, multiplier=3.0):
    high, low, close = df["High"], df["Low"], df["Close"]
    atr = calculate_atr(high, low, close, window=period)
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    upper_band = basic_upper.copy()
    lower_band = basic_lower.copy()
    
    for i in range(1, len(df)):
        if basic_upper.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = basic_upper.iloc[i]
        else: 
            upper_band.iloc[i] = upper_band.iloc[i-1]
        if basic_lower.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = basic_lower.iloc[i]
        else: 
            lower_band.iloc[i] = lower_band.iloc[i-1]
            
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if direction.iloc[i-1] == 1:
            if close.iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
        else:
            if close.iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
    return supertrend, direction

def compute_all_indicators(df: pd.DataFrame) -> dict:
    """Compute short-term momentum indicators"""
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    
    sma50 = calculate_sma(close, window=50)
    trend_verdict = "Bullish" if close.iloc[-1] > sma50.iloc[-1] else "Bearish"
    rsi = calculate_rsi(close, window=14)
    macd_hist = calculate_macd(close)
    roc = ((close - close.shift(10)) / close.shift(10) * 100).fillna(0)
    stoch = calculate_stochastic(high, low, close, window=14, smooth=3)
    williams_r = calculate_williams_r(high, low, close, window=14)
    cci = calculate_cci(high, low, close, window=20)
    adx = calculate_adx(high, low, close, window=14)
    atr = calculate_atr(high, low, close, window=14)
    obv = calculate_obv(close, volume)
    vwap = calculate_vwap(high, low, close, volume, window=20)
    
    bb_high, bb_low = calculate_bollinger_bands(close, window=20, num_std=2)
    bb_verdict = "Inside Bands"
    if close.iloc[-1] > bb_high.iloc[-1]: 
        bb_verdict = "Overbought"
    elif close.iloc[-1] < bb_low.iloc[-1]: 
        bb_verdict = "Oversold"
    
    _, st_dir = calculate_supertrend(df)
    st_verdict = "BUY" if st_dir.iloc[-1] == 1 else "SELL"
    
    high_52w = high.rolling(window=252, min_periods=1).max().iloc[-1]
    low_52w = low.rolling(window=252, min_periods=1).min().iloc[-1]
    
    return {
        "price": close.iloc[-1], "trend": trend_verdict, "rsi": rsi.iloc[-1],
        "macd_hist": macd_hist.iloc[-1], "roc": roc.iloc[-1], "stoch": stoch.iloc[-1],
        "williams_r": williams_r.iloc[-1], "cci": cci.iloc[-1], "adx": adx.iloc[-1],
        "atr": atr.iloc[-1], "obv": obv.iloc[-1], "vwap": vwap.iloc[-1],
        "bollinger": bb_verdict, "supertrend": st_verdict, "volume": volume.iloc[-1],
        "52w_high_pct": ((close.iloc[-1] - high_52w) / high_52w) * 100,
        "52w_low_pct": ((close.iloc[-1] - low_52w) / low_52w) * 100
    }

def compute_long_term_indicators(df: pd.DataFrame) -> dict:
    """Compute long-term trend indicators"""
    close = df["Close"]
    
    sma200 = calculate_sma(close, window=200)
    long_term_trend = "Bullish" if close.iloc[-1] > sma200.iloc[-1] else "Bearish"
    
    annual_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    
    high_52w = close.rolling(window=252, min_periods=1).max().iloc[-1]
    low_52w = close.rolling(window=252, min_periods=1).min().iloc[-1]
    support = low_52w
    resistance = high_52w
    distance_to_support = ((close.iloc[-1] - support) / support) * 100
    
    # 3-Year trend
    if len(df) >= 756:
        sma_3y = calculate_sma(close, window=756)
        trend_3y = "Bullish" if close.iloc[-1] > sma_3y.iloc[-1] else "Bearish"
        return_3y = ((close.iloc[-1] - close.iloc[-756]) / close.iloc[-756]) * 100
    else:
        trend_3y = long_term_trend
        return_3y = annual_return
    
    return {
        "sma200": sma200.iloc[-1],
        "long_term_trend": long_term_trend,
        "annual_return": annual_return,
        "support": support,
        "resistance": resistance,
        "distance_to_support": distance_to_support,
        "trend_3y": trend_3y,
        "return_3y": return_3y
    }


# =========================================================================
# NEWS SENTIMENT & SCORING MATRICES
# =========================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_google_news(company_name: str):
    url = f"https://news.google.com/rss/search?q={quote(f'{company_name} 주가')}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        item = ET.fromstring(resp.content).find(".//item")
        if item is not None: 
            return item.findtext("title"), item.findtext("link"), "Google News KR"
    except: 
        pass
    return "No recent headline found", "#", "N/A"

def process_scoring(ind: dict, long_term: dict):
    """Generate short-term, long-term, and ultra long-term recommendations"""
    
    # SHORT-TERM MOMENTUM SCORE
    score = 50.0
    if ind["rsi"] < 30: 
        score += 15
    elif ind["rsi"] > 70: 
        score -= 15
    if ind["macd_hist"] > 0: 
        score += 10
    else: 
        score -= 5
    if ind["supertrend"] == "BUY": 
        score += 10
    else: 
        score -= 10
    if ind["cci"] > 100: 
        score += 5
    elif ind["cci"] < -100: 
        score -= 5
    
    score = max(0.0, min(100.0, score))
    rec = "HOLD"
    if score >= 65: 
        rec = "BUY"
    elif score <= 35: 
        rec = "SELL"
    
    # LONG-TERM (1-YEAR) STRATEGIC SCORE
    lt_score = 50.0
    if long_term["long_term_trend"] == "Bullish": 
        lt_score += 25
    else: 
        lt_score -= 20
    
    if long_term["annual_return"] > 15: 
        lt_score += 15
    elif long_term["annual_return"] < -10: 
        lt_score -= 15
    
    if long_term["distance_to_support"] < 5:
        lt_score += 20
    
    lt_score = max(0.0, min(100.0, lt_score))
    lt_rec = "HOLD"
    if lt_score >= 65: 
        lt_rec = "BUY"
    elif lt_score <= 35: 
        lt_rec = "SELL"
    
    # ULTRA LONG-TERM (3-YEAR) STRATEGIC SCORE
    ult_score = 50.0
    if long_term["trend_3y"] == "Bullish": 
        ult_score += 35
    else: 
        ult_score -= 25
    
    if long_term["return_3y"] > 30: 
        ult_score += 25
    elif long_term["return_3y"] > 10: 
        ult_score += 15
    elif long_term["return_3y"] < -20: 
        ult_score -= 25
    
    if long_term["distance_to_support"] < 10:
        ult_score += 15
    elif long_term["distance_to_support"] > 20:
        ult_score -= 10
    
    ult_score = max(0.0, min(100.0, ult_score))
    ult_rec = "HOLD"
    if ult_score >= 65: 
        ult_rec = "BUY"
    elif ult_score <= 35: 
        ult_rec = "SELL"
    
    # COMBINED VERDICT LOGIC - LONG-TERM BUY RECOMMENDATIONS
    if ult_rec == "BUY" and (rec == "SELL" or lt_rec == "SELL"):
        combined_verdict = "💎 ACCUMULATE (3Y Bullish)"
        absolute_rec = "BUY (Long-term Hold)"
    elif ult_rec == "BUY" and lt_rec == "BUY" and rec == "BUY":
        combined_verdict = "🚀 STRONG BUY (All Signals)"
        absolute_rec = "BUY (All Timeframes)"
    elif rec == "SELL" and lt_rec == "BUY" and ult_rec == "BUY":
        combined_verdict = "🔄 BUY on Dips (LT+ULT Bullish)"
        absolute_rec = "BUY (On Pullback)"
    elif rec == "BUY" and lt_rec == "SELL" and ult_rec == "BUY":
        combined_verdict = "⚠️ CAUTIOUS BUY (ULT Bullish)"
        absolute_rec = "HOLD (Take Profits)"
    elif ult_rec == "SELL":
        combined_verdict = "🔴 AVOID (ULT Bearish)"
        absolute_rec = "SELL (Downtrend)"
    else:
        combined_verdict = f"→ {rec}"
        absolute_rec = rec
    
    return {
        "short_term_score": round(score, 1),
        "short_term_rec": rec,
        "long_term_score": round(lt_score, 1),
        "long_term_rec": lt_rec,
        "ultra_long_term_score": round(ult_score, 1),
        "ultra_long_term_rec": ult_rec,
        "combined_verdict": combined_verdict,
        "absolute_rec": absolute_rec
    }


# =========================================================================
# WORKER ENGINES
# =========================================================================
def scan_single_ticker(ticker: str, name: str):
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="3y")
        
        df = df.dropna(subset=["Close", "High", "Low", "Volume"])
        
        if df.empty or len(df) < 50: 
            return None
        
        try:
            info = ticker_obj.info
            if info is None:
                info = {}
        except:
            info = {}
        
        pe_formatted = calculate_pe_ratio(ticker, info, df)
        
        mcap = info.get("marketCap")
        if mcap is None or pd.isna(mcap):
            mcap = "N/A"
        else:
            try:
                mcap = f"{int(mcap):,}"
            except:
                mcap = "N/A"
        
        latest = compute_all_indicators(df)
        long_term = compute_long_term_indicators(df)
        
        headline, link, source = get_google_news(name)
        scoring = process_scoring(latest, long_term)
        
        price_val = latest["price"]
        price_display = f"₩{price_val:,.0f}" if not pd.isna(price_val) else "N/A"
        
        return {
            "Stock": f"{name} ({ticker})", 
            "ST Score": scoring["short_term_score"],
            "ST Rec": scoring["short_term_rec"],
            "LT Score": scoring["long_term_score"],
            "LT Rec": scoring["long_term_rec"],
            "ULT Score": scoring["ultra_long_term_score"],
            "ULT Rec": scoring["ultra_long_term_rec"],
            "Combined Strategy": scoring["combined_verdict"],
            "Action": scoring["absolute_rec"],
            "Current Price": price_display,
            "LT Trend": long_term["long_term_trend"],
            "3Y Trend": long_term["trend_3y"],
            "Annual Return": f"{long_term['annual_return']:.1f}%",
            "3-Year Return": f"{long_term['return_3y']:.1f}%",
            "Support Level": f"₩{long_term['support']:,.0f}",
            "Resistance": f"₩{long_term['resistance']:,.0f}",
            "Distance to Support": f"{long_term['distance_to_support']:.1f}%",
            "Trend (MA)": latest["trend"], 
            "Market Cap": mcap, 
            "P/E Ratio": pe_formatted,
            "Sentiment": "Neutral",
            "Headline": headline, 
            "Source": source,
            "Supertrend": latest["supertrend"], 
            "MACD": round(latest["macd_hist"], 2), 
            "RSI": round(latest["rsi"], 1), 
            "ROC(10d)": round(latest["roc"], 2),
            "Stochastic": round(latest["stoch"], 1), 
            "Williams %R": round(latest["williams_r"], 1), 
            "CCI": round(latest["cci"], 1),
            "ADX": round(latest["adx"], 1), 
            "OBV": f"{latest['obv']:,.0f}", 
            "VWAP(20d)": round(latest["vwap"], 0),
            "Bollinger": latest["bollinger"], 
            "ATR": round(latest["atr"], 0),
            "52W High %": round(latest["52w_high_pct"], 2), 
            "52W Low %": round(latest["52w_low_pct"], 2), 
            "Volume": f"{latest['volume']:,.0f}"
        }
    except Exception as e:
        return None

def run_parallel_scan(tickers_dict: dict, max_workers: int = 8, progress_callback=None):
    results = []
    total = len(tickers_dict)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_single_ticker, t, n): (t, n) for t, n in tickers_dict.items()}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if progress_callback: 
                progress_callback(done, total)
            res = future.result()
            if res: 
                results.append(res)
    return results


# =========================================================================
# COLOR STYLING FUNCTION
# =========================================================================
def highlight_recommendations(val):
    """Color code recommendations"""
    if isinstance(val, str):
        if "STRONG BUY" in val or "ACCUMULATE" in val:
            return 'background-color: #1e5631; color: #ffffff; font-weight: bold;'
        elif "BUY on Dips" in val:
            return 'background-color: #cce5ff; color: #004085; font-weight: bold;'
        elif "BUY" in val.upper() and "REC" not in val and "STRONG" not in val:
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif "AVOID" in val:
            return 'background-color: #8b0000; color: #ffffff; font-weight: bold;'
        elif "SELL" in val.upper():
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        elif "HOLD" in val.upper() or "CAUTIOUS" in val:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
    return ''


# =========================================================================
# APPLICATION UI
# =========================================================================
def main():
    st.set_page_config(page_title="KOSPI 200 Advanced Screener", layout="wide")
    st.title("🇰🇷 KOSPI 200 Tri-Timeframe Momentum Screener")
    st.caption("Short-term momentum + Long-term trends + Ultra long-term (3Y) accumulation signals")

    tickers_all = get_kospi200_tickers()
    
    with st.sidebar:
        st.header("⚙️ Scanning Framework")
        subset_n = st.slider("Universe Depth Scan Size", 5, len(tickers_all), min(38, len(tickers_all)))
        max_workers = st.slider("Parallel Threads Execution", 2, 16, 8)
        
        run_btn = st.button("🔍 Initialize Deep Stock Scanning Engine", type="primary", use_container_width=True)

    if "scan_data" not in st.session_state:
        st.session_state["scan_data"] = None

    if run_btn:
        subset = dict(list(tickers_all.items())[:subset_n])
        progress = st.progress(0.0, text="Initializing scan...")
        
        def _cb(d, t):
            progress.progress(d / t, text=f"Scanned {d}/{t} tickers...")
        
        with st.spinner("Processing tri-timeframe matrix transformations..."):
            results = run_parallel_scan(subset, max_workers=max_workers, progress_callback=_cb)
        
        progress.empty()
        st
