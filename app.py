import math
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

RISK_FREE_RATE = 7.0  # Approx. Indian 10Y G-Sec yield, used for Sharpe ratio

# =========================================================================
# NIFTY 50 UNIVERSE (verified current constituents, with sector tags)
# Sector tags are used to apply sector-aware fundamental scoring
# (e.g. banks/NBFCs are structurally high-leverage, so debt/equity is not
# penalized the same way it would be for an industrial company).
# =========================================================================
FALLBACK_NIFTY50 = {
    "ADANIENT.NS":    ("Adani Enterprises", "Metals & Mining"),
    "ADANIPORTS.NS":  ("Adani Ports & SEZ", "Services"),
    "APOLLOHOSP.NS":  ("Apollo Hospitals", "Healthcare"),
    "ASIANPAINT.NS":  ("Asian Paints", "Consumer Durables"),
    "AXISBANK.NS":    ("Axis Bank", "Financial Services"),
    "BAJAJ-AUTO.NS":  ("Bajaj Auto", "Automobile"),
    "BAJFINANCE.NS":  ("Bajaj Finance", "Financial Services"),
    "BAJAJFINSV.NS":  ("Bajaj Finserv", "Financial Services"),
    "BEL.NS":         ("Bharat Electronics", "Capital Goods"),
    "BHARTIARTL.NS":  ("Bharti Airtel", "Telecommunication"),
    "CIPLA.NS":       ("Cipla", "Healthcare"),
    "COALINDIA.NS":   ("Coal India", "Oil Gas & Consumable Fuels"),
    "DRREDDY.NS":     ("Dr. Reddy's Laboratories", "Healthcare"),
    "EICHERMOT.NS":   ("Eicher Motors", "Automobile"),
    "ETERNAL.NS":     ("Eternal (Zomato)", "Consumer Services"),
    "GRASIM.NS":      ("Grasim Industries", "Construction Materials"),
    "HCLTECH.NS":     ("HCL Technologies", "Information Technology"),
    "HDFCBANK.NS":    ("HDFC Bank", "Financial Services"),
    "HDFCLIFE.NS":    ("HDFC Life Insurance", "Financial Services"),
    "HINDALCO.NS":    ("Hindalco Industries", "Metals & Mining"),
    "HINDUNILVR.NS":  ("Hindustan Unilever", "FMCG"),
    "ICICIBANK.NS":   ("ICICI Bank", "Financial Services"),
    "INDIGO.NS":      ("InterGlobe Aviation (IndiGo)", "Services"),
    "INFY.NS":        ("Infosys", "Information Technology"),
    "ITC.NS":         ("ITC Limited", "FMCG"),
    "JIOFIN.NS":      ("Jio Financial Services", "Financial Services"),
    "JSWSTEEL.NS":    ("JSW Steel", "Metals & Mining"),
    "KOTAKBANK.NS":   ("Kotak Mahindra Bank", "Financial Services"),
    "LT.NS":          ("Larsen & Toubro", "Construction"),
    "M&M.NS":         ("Mahindra & Mahindra", "Automobile"),
    "MARUTI.NS":      ("Maruti Suzuki", "Automobile"),
    "MAXHEALTH.NS":   ("Max Healthcare", "Healthcare"),
    "NESTLEIND.NS":   ("Nestle India", "FMCG"),
    "NTPC.NS":        ("NTPC Limited", "Power"),
    "ONGC.NS":        ("Oil & Natural Gas Corp", "Oil Gas & Consumable Fuels"),
    "POWERGRID.NS":   ("Power Grid Corporation", "Power"),
    "RELIANCE.NS":    ("Reliance Industries", "Oil Gas & Consumable Fuels"),
    "SBILIFE.NS":     ("SBI Life Insurance", "Financial Services"),
    "SHRIRAMFIN.NS":  ("Shriram Finance", "Financial Services"),
    "SBIN.NS":        ("State Bank of India", "Financial Services"),
    "SUNPHARMA.NS":   ("Sun Pharmaceutical", "Healthcare"),
    "TCS.NS":         ("Tata Consultancy Services", "Information Technology"),
    "TATACONSUM.NS":  ("Tata Consumer Products", "FMCG"),
    "TMPV.NS":        ("Tata Motors Passenger Vehicles", "Automobile"),
    "TATASTEEL.NS":   ("Tata Steel", "Metals & Mining"),
    "TECHM.NS":       ("Tech Mahindra", "Information Technology"),
    "TITAN.NS":       ("Titan Company", "Consumer Durables"),
    "TRENT.NS":       ("Trent", "Consumer Services"),
    "ULTRACEMCO.NS":  ("UltraTech Cement", "Construction Materials"),
    "WIPRO.NS":       ("Wipro", "Information Technology"),
}

FINANCIAL_SECTOR = "Financial Services"


# =========================================================================
# BUILT-IN TECHNICAL INDICATOR CALCULATIONS
# =========================================================================
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd_hist.fillna(0)

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.fillna(0)

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def calculate_stochastic(high, low, close, window=14, smooth=3):
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k.rolling(window=smooth).mean()
    return k.fillna(50)

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci.fillna(0)

def calculate_adx(high, low, close, window=14):
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
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def calculate_vwap(high, low, close, volume, window=20):
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap.fillna(close)

def calculate_williams_r(high, low, close, window=14):
    highest = high.rolling(window=window).max()
    lowest = low.rolling(window=window).min()
    wr = -100 * (highest - close) / (highest - lowest)
    return wr.fillna(-50)

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty50_tickers():
    return dict(FALLBACK_NIFTY50)


# =========================================================================
# TECHNICAL INDICATOR AGGREGATION
# =========================================================================
def compute_all_indicators(df: pd.DataFrame) -> dict:
    """Short-term momentum indicators (uses recent price action)."""
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
    """Long-term (1Y / 3Y) price trend indicators."""
    close = df["Close"]
    sma200 = calculate_sma(close, window=200)
    long_term_trend = "Bullish" if close.iloc[-1] > sma200.iloc[-1] else "Bearish"
    annual_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100

    high_52w = close.rolling(window=252, min_periods=1).max().iloc[-1]
    low_52w = close.rolling(window=252, min_periods=1).min().iloc[-1]
    support = low_52w
    resistance = high_52w
    distance_to_support = ((close.iloc[-1] - support) / support) * 100

    if len(df) >= 756:
        sma_3y = calculate_sma(close, window=756)
        trend_3y = "Bullish" if close.iloc[-1] > sma_3y.iloc[-1] else "Bearish"
        return_3y = ((close.iloc[-1] - close.iloc[-756]) / close.iloc[-756]) * 100
    else:
        trend_3y = long_term_trend
        return_3y = annual_return

    return {
        "sma200": sma200.iloc[-1], "long_term_trend": long_term_trend,
        "annual_return": annual_return, "support": support, "resistance": resistance,
        "distance_to_support": distance_to_support, "trend_3y": trend_3y, "return_3y": return_3y
    }


def compute_risk_metrics(df: pd.DataFrame, info: dict) -> dict:
    """Volatility, drawdown, Sharpe ratio, and beta — how bumpy the ride is."""
    close = df["Close"]
    daily_returns = close.pct_change().dropna()

    annual_vol = daily_returns.std() * math.sqrt(252) * 100 if len(daily_returns) > 1 else np.nan

    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else np.nan

    years = len(df) / 252
    if years > 0 and close.iloc[0] > 0:
        cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100
    else:
        cagr = np.nan

    if annual_vol and not np.isnan(annual_vol) and annual_vol > 0 and not np.isnan(cagr):
        sharpe = (cagr - RISK_FREE_RATE) / annual_vol
    else:
        sharpe = np.nan

    beta = info.get("beta")
    try:
        beta = float(beta) if beta is not None else np.nan
    except (TypeError, ValueError):
        beta = np.nan

    if np.isnan(annual_vol):
        risk_grade = "N/A"
    elif annual_vol < 20:
        risk_grade = "Low"
    elif annual_vol < 30:
        risk_grade = "Moderate"
    elif annual_vol < 45:
        risk_grade = "High"
    else:
        risk_grade = "Very High"

    return {
        "annual_volatility": annual_vol,
        "max_drawdown": max_drawdown,
        "cagr_3y": cagr,
        "sharpe_ratio": sharpe,
        "beta": beta,
        "risk_grade": risk_grade
    }


# =========================================================================
# FUNDAMENTAL ANALYSIS ENGINE
# =========================================================================
def _safe_pct(val):
    """yfinance sometimes returns fractions (0.15) and sometimes already-percentages.
    Normalize to a percentage float, or None if unavailable."""
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val * 100 if abs(val) < 5 else val


def get_fundamentals(ticker: str, info: dict, sector: str) -> dict:
    """Pull the fundamental datapoints that actually matter for a long-term
    quality assessment, rather than relying on price trend as a proxy."""
    is_financial = sector == FINANCIAL_SECTOR

    roe = _safe_pct(info.get("returnOnEquity"))
    profit_margin = _safe_pct(info.get("profitMargins"))
    revenue_growth = _safe_pct(info.get("revenueGrowth"))
    earnings_growth = _safe_pct(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
    debt_to_equity = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    dividend_yield = _safe_pct(info.get("dividendYield"))
    price_to_book = info.get("priceToBook")
    peg_ratio = info.get("pegRatio") or info.get("trailingPegRatio")

    try:
        debt_to_equity = float(debt_to_equity) if debt_to_equity is not None else None
    except (TypeError, ValueError):
        debt_to_equity = None
    try:
        current_ratio = float(current_ratio) if current_ratio is not None else None
    except (TypeError, ValueError):
        current_ratio = None
    try:
        price_to_book = float(price_to_book) if price_to_book is not None else None
    except (TypeError, ValueError):
        price_to_book = None
    try:
        peg_ratio = float(peg_ratio) if peg_ratio is not None else None
    except (TypeError, ValueError):
        peg_ratio = None

    return {
        "roe": roe,
        "profit_margin": profit_margin,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "dividend_yield": dividend_yield,
        "price_to_book": price_to_book,
        "peg_ratio": peg_ratio,
        "is_financial": is_financial,
    }


def compute_fundamental_score(f: dict) -> dict:
    """
    Quality score (0-100) built from profitability, growth, leverage,
    liquidity and valuation. Debt/equity and current ratio are skipped for
    banks/NBFCs/insurers, since high leverage is structural to their
    business model (deposits and policy liabilities are not comparable to
    industrial company debt).
    """
    score = 50.0
    data_points_found = 0
    notes = []

    if f["roe"] is not None:
        data_points_found += 1
        if f["roe"] >= 20:
            score += 20; notes.append(f"Strong ROE ({f['roe']:.1f}%)")
        elif f["roe"] >= 15:
            score += 12
        elif f["roe"] >= 10:
            score += 3
        elif f["roe"] >= 0:
            score -= 8
        else:
            score -= 20; notes.append("Negative ROE")

    if f["profit_margin"] is not None:
        data_points_found += 1
        if f["profit_margin"] >= 20:
            score += 12; notes.append(f"High margins ({f['profit_margin']:.1f}%)")
        elif f["profit_margin"] >= 10:
            score += 6
        elif f["profit_margin"] >= 0:
            score += 0
        else:
            score -= 15; notes.append("Negative margins")

    if f["revenue_growth"] is not None:
        data_points_found += 1
        if f["revenue_growth"] >= 15:
            score += 8
        elif f["revenue_growth"] >= 5:
            score += 4
        elif f["revenue_growth"] < 0:
            score -= 8

    if f["earnings_growth"] is not None:
        data_points_found += 1
        if f["earnings_growth"] >= 15:
            score += 8
        elif f["earnings_growth"] >= 5:
            score += 4
        elif f["earnings_growth"] < 0:
            score -= 10; notes.append("Declining earnings")

    if not f["is_financial"]:
        if f["debt_to_equity"] is not None:
            data_points_found += 1
            if f["debt_to_equity"] < 50:
                score += 10; notes.append("Low leverage")
            elif f["debt_to_equity"] < 100:
                score += 4
            elif f["debt_to_equity"] < 200:
                score -= 5
            else:
                score -= 15; notes.append("High leverage")

        if f["current_ratio"] is not None:
            data_points_found += 1
            if f["current_ratio"] >= 1.5:
                score += 5
            elif f["current_ratio"] < 1.0:
                score -= 8; notes.append("Weak liquidity")
    else:
        # For financials, profitability & growth quality carry the extra weight
        # that would otherwise go to leverage/liquidity checks.
        if f["roe"] is not None and f["roe"] >= 15:
            score += 5

    if f["peg_ratio"] is not None and f["peg_ratio"] > 0:
        data_points_found += 1
        if f["peg_ratio"] < 1:
            score += 8; notes.append("Attractively priced vs growth")
        elif f["peg_ratio"] < 2:
            score += 2
        else:
            score -= 8; notes.append("Expensive vs growth")

    if f["dividend_yield"] is not None and f["dividend_yield"] >= 1.5:
        score += 3

    score = max(0.0, min(100.0, score))

    if data_points_found < 2:
        rating = "Insufficient Data"
    elif score >= 75:
        rating = "Excellent"
    elif score >= 60:
        rating = "Good"
    elif score >= 40:
        rating = "Average"
    elif score >= 25:
        rating = "Weak"
    else:
        rating = "Poor"

    return {
        "fundamental_score": round(score, 1),
        "fundamental_rating": rating,
        "fundamental_notes": "; ".join(notes) if notes else "Limited data available",
        "data_points": data_points_found
    }


def calculate_pe_ratio(info: dict, df: pd.DataFrame) -> str:
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
            if 0 < pe < 1000:
                return f"{pe:.2f}"
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return "N/A"


@st.cache_data(ttl=1800, show_spinner=False)
def get_google_news(company_name: str):
    url = f"https://news.google.com/rss/search?q={quote(f'{company_name} stock')}&hl=en&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        item = ET.fromstring(resp.content).find(".//item")
        if item is not None:
            return item.findtext("title"), item.findtext("link"), "Google News"
    except Exception:
        pass
    return "No recent headline found", "#", "N/A"


# =========================================================================
# COMBINED SCORING — Technical (short & long term) + Fundamental + Risk
# =========================================================================
def process_scoring(ind: dict, long_term: dict, fund_score: dict, risk: dict) -> dict:
    # --- Short-term momentum score (unchanged logic) ---
    score = 50.0
    if ind["rsi"] < 30: score += 15
    elif ind["rsi"] > 70: score -= 15
    if ind["macd_hist"] > 0: score += 10
    else: score -= 5
    if ind["supertrend"] == "BUY": score += 10
    else: score -= 10
    if ind["cci"] > 100: score += 5
    elif ind["cci"] < -100: score -= 5
    score = max(0.0, min(100.0, score))
    rec = "HOLD"
    if score >= 65: rec = "BUY"
    elif score <= 35: rec = "SELL"

    # --- Long-term (1Y) price-trend score ---
    lt_score = 50.0
    if long_term["long_term_trend"] == "Bullish": lt_score += 25
    else: lt_score -= 20
    if long_term["annual_return"] > 15: lt_score += 15
    elif long_term["annual_return"] < -10: lt_score -= 15
    if long_term["distance_to_support"] < 5: lt_score += 20
    lt_score = max(0.0, min(100.0, lt_score))
    lt_rec = "HOLD"
    if lt_score >= 65: lt_rec = "BUY"
    elif lt_score <= 35: lt_rec = "SELL"

    # --- 3-year price trend score (ultra long-term technical) ---
    ult_score = 50.0
    if long_term["trend_3y"] == "Bullish": ult_score += 20
    else: ult_score -= 15
    if long_term["return_3y"] > 30: ult_score += 15
    elif long_term["return_3y"] > 10: ult_score += 8
    elif long_term["return_3y"] < -20: ult_score -= 20
    ult_score = max(0.0, min(100.0, ult_score))

    # --- Fundamental quality score (the actual "is this a good business" check) ---
    f_score = fund_score["fundamental_score"]
    f_rating = fund_score["fundamental_rating"]

    # --- Risk-adjusted modifier ---
    risk_penalty = 0.0
    if not np.isnan(risk.get("annual_volatility", np.nan)):
        if risk["annual_volatility"] > 45: risk_penalty -= 10
        elif risk["annual_volatility"] > 30: risk_penalty -= 4
    if not np.isnan(risk.get("sharpe_ratio", np.nan)):
        if risk["sharpe_ratio"] > 0.5: risk_penalty += 8
        elif risk["sharpe_ratio"] < -0.3: risk_penalty -= 8

    # --- Composite overall investment score ---
    # Weighted: fundamentals matter most for a long-term view; short-term
    # momentum matters least. Weights: ST 15%, LT(1Y) 20%, ULT(3Y) 15%,
    # Fundamentals 35%, Risk-adjustment 15%.
    overall = (0.15 * score) + (0.20 * lt_score) + (0.15 * ult_score) + \
              (0.35 * f_score) + (0.15 * (50 + risk_penalty))
    overall = max(0.0, min(100.0, overall))

    fundamentally_strong = f_rating in ("Excellent", "Good")
    fundamentally_weak = f_rating in ("Weak", "Poor")
    technically_strong = rec == "BUY" and lt_rec == "BUY"
    technically_weak = rec == "SELL" or lt_rec == "SELL"

    if fundamentally_strong and technically_strong:
        combined_verdict = "🏆 QUALITY BUY (Strong Fundamentals + Momentum)"
        absolute_rec = "BUY (High Conviction)"
    elif fundamentally_strong and technically_weak:
        combined_verdict = "💎 ACCUMULATE (Quality Compounder, Price Weak Now)"
        absolute_rec = "BUY (Long-term Accumulation)"
    elif fundamentally_strong and not technically_strong and not technically_weak:
        combined_verdict = "✅ HOLD / ADD ON DIPS (Solid Fundamentals)"
        absolute_rec = "HOLD (Add on Weakness)"
    elif fundamentally_weak and technically_strong:
        combined_verdict = "⚠️ SPECULATIVE (Momentum without Fundamentals)"
        absolute_rec = "CAUTION (Momentum Trade Only)"
    elif fundamentally_weak:
        combined_verdict = "🔴 AVOID (Weak Fundamentals)"
        absolute_rec = "SELL / AVOID"
    else:
        combined_verdict = f"→ {rec} (Data-Limited Fundamentals)"
        absolute_rec = rec

    return {
        "short_term_score": round(score, 1), "short_term_rec": rec,
        "long_term_score": round(lt_score, 1), "long_term_rec": lt_rec,
        "fundamental_score": f_score, "fundamental_rating": f_rating,
        "overall_score": round(overall, 1),
        "combined_verdict": combined_verdict,
        "absolute_rec": absolute_rec
    }


# =========================================================================
# WORKER ENGINES
# =========================================================================
def scan_single_ticker(ticker: str, name: str, sector: str):
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
        except Exception:
            info = {}

        pe_formatted = calculate_pe_ratio(info, df)

        mcap = info.get("marketCap")
        if mcap is None or pd.isna(mcap):
            mcap_display = "N/A"
        else:
            try:
                mcap_display = f"₹{int(mcap/10000000):,}Cr"
            except Exception:
                mcap_display = "N/A"

        latest = compute_all_indicators(df)
        long_term = compute_long_term_indicators(df)
        risk = compute_risk_metrics(df, info)
        fundamentals = get_fundamentals(ticker, info, sector)
        fund_score = compute_fundamental_score(fundamentals)
        headline, link, source = get_google_news(name)
        scoring = process_scoring(latest, long_term, fund_score, risk)

        price_val = latest["price"]
        price_display = f"₹{price_val:,.2f}" if not pd.isna(price_val) else "N/A"

        def fmt_pct(v, decimals=1):
            return f"{v:.{decimals}f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"

        def fmt_num(v, decimals=2):
            return f"{v:.{decimals}f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"

        return {
            "Stock": f"{name} ({ticker})",
            "Sector": sector,
            "Overall Score": scoring["overall_score"],
            "Combined Strategy": scoring["combined_verdict"],
            "Action": scoring["absolute_rec"],
            "Fundamental Score": scoring["fundamental_score"],
            "Fundamental Rating": scoring["fundamental_rating"],
            "Fund. Notes": fund_score["fundamental_notes"],
            "ST Score": scoring["short_term_score"], "ST Rec": scoring["short_term_rec"],
            "LT Score": scoring["long_term_score"], "LT Rec": scoring["long_term_rec"],
            "Current Price": price_display,
            "LT Trend": long_term["long_term_trend"], "3Y Trend": long_term["trend_3y"],
            "Annual Return": f"{long_term['annual_return']:.1f}%",
            "3-Year Return": f"{long_term['return_3y']:.1f}%",
            "Support Level": f"₹{long_term['support']:,.2f}",
            "Resistance": f"₹{long_term['resistance']:,.2f}",
            "Distance to Support": f"{long_term['distance_to_support']:.1f}%",
            "Market Cap": mcap_display, "P/E Ratio": pe_formatted,
            "ROE": fmt_pct(fundamentals["roe"]),
            "Profit Margin": fmt_pct(fundamentals["profit_margin"]),
            "Revenue Growth": fmt_pct(fundamentals["revenue_growth"]),
            "Earnings Growth": fmt_pct(fundamentals["earnings_growth"]),
            "Debt/Equity": fmt_num(fundamentals["debt_to_equity"], 0) if not fundamentals["is_financial"] else "N/A (Financial)",
            "P/B Ratio": fmt_num(fundamentals["price_to_book"]),
            "PEG Ratio": fmt_num(fundamentals["peg_ratio"]),
            "Dividend Yield": fmt_pct(fundamentals["dividend_yield"]),
            "Volatility (Ann.)": fmt_pct(risk["annual_volatility"]),
            "Max Drawdown (3Y)": fmt_pct(risk["max_drawdown"]),
            "Sharpe Ratio": fmt_num(risk["sharpe_ratio"]),
            "Beta": fmt_num(risk["beta"]),
            "Risk Grade": risk["risk_grade"],
            "Headline": headline, "Source": source,
            "Supertrend": latest["supertrend"], "MACD": round(latest["macd_hist"], 2),
            "RSI": round(latest["rsi"], 1), "ROC(10d)": round(latest["roc"], 2),
            "Stochastic": round(latest["stoch"], 1), "Williams %R": round(latest["williams_r"], 1),
            "CCI": round(latest["cci"], 1), "ADX": round(latest["adx"], 1),
            "OBV": f"{latest['obv']:,.0f}", "VWAP(20d)": round(latest["vwap"], 2),
            "Bollinger": latest["bollinger"], "ATR": round(latest["atr"], 2),
            "52W High %": round(latest["52w_high_pct"], 2),
            "52W Low %": round(latest["52w_low_pct"], 2),
            "Volume": f"{latest['volume']:,.0f}"
        }
    except Exception:
        return None


def run_parallel_scan(tickers_dict: dict, max_workers: int = 8, progress_callback=None):
    results = []
    total = len(tickers_dict)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_single_ticker, t, meta[0], meta[1]): t
            for t, meta in tickers_dict.items()
        }
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if progress_callback:
                progress_callback(done, total)
            res = future.result()
            if res:
                results.append(res)
    return results


# =========================================================================
# COLOR STYLING
# =========================================================================
def highlight_recommendations(val):
    if isinstance(val, str):
        if "QUALITY BUY" in val:
            return 'background-color: #1e5631; color: #ffffff; font-weight: bold;'
        elif "ACCUMULATE" in val:
            return 'background-color: #cce5ff; color: #004085; font-weight: bold;'
        elif "HOLD / ADD" in val:
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif "SPECULATIVE" in val:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
        elif "AVOID" in val:
            return 'background-color: #8b0000; color: #ffffff; font-weight: bold;'
        elif "BUY" in val.upper():
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif "SELL" in val.upper():
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        elif "HOLD" in val.upper() or "CAUTION" in val.upper():
            return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
    return ''


def highlight_fundamental_rating(val):
    mapping = {
        "Excellent": 'background-color: #1e5631; color: #ffffff; font-weight: bold;',
        "Good": 'background-color: #d4edda; color: #155724; font-weight: bold;',
        "Average": 'background-color: #fff3cd; color: #856404; font-weight: bold;',
        "Weak": 'background-color: #f8d7da; color: #721c24; font-weight: bold;',
        "Poor": 'background-color: #8b0000; color: #ffffff; font-weight: bold;',
    }
    return mapping.get(val, '')


# =========================================================================
# APPLICATION UI
# =========================================================================
def main():
    st.set_page_config(page_title="Nifty 50 Quality Screener", layout="wide")
    st.title("🇮🇳 Nifty 50 Fundamental + Technical Screener")
    st.caption(
        "Combines profitability, growth, leverage & valuation (fundamentals) with "
        "short/long-term price trends and risk metrics — so quality and momentum "
        "are scored separately, not conflated."
    )
    st.info(
        "⚠️ **Educational tool, not financial advice.** Scores are derived from "
        "automated calculations on data pulled from Yahoo Finance, which can be "
        "incomplete or delayed for NSE-listed stocks. Verify anything material "
        "before acting on it, and consider consulting a licensed financial advisor.",
        icon="⚠️"
    )

    tickers_all = get_nifty50_tickers()
    all_sectors = sorted(set(v[1] for v in tickers_all.values()))

    with st.sidebar:
        st.header("⚙️ Scanning Framework")
        subset_n = st.slider("Universe Depth Scan Size", 5, len(tickers_all), len(tickers_all))
        max_workers = st.slider("Parallel Threads Execution", 2, 16, 8)
        sector_filter_pre = st.multiselect("Limit scan to sectors (optional)", options=all_sectors, default=[])
        run_btn = st.button("🔍 Run Full Analysis", type="primary", use_container_width=True)

    if "scan_data" not in st.session_state:
        st.session_state["scan_data"] = None

    if run_btn:
        universe = tickers_all
        if sector_filter_pre:
            universe = {t: m for t, m in tickers_all.items() if m[1] in sector_filter_pre}
        subset = dict(list(universe.items())[:subset_n])

        progress = st.progress(0.0, text="Initializing scan...")

        def _cb(d, t):
            progress.progress(d / t, text=f"Analyzing {d}/{t} stocks (technicals + fundamentals + risk)...")

        with st.spinner("Running fundamental, technical & risk analysis..."):
            results = run_parallel_scan(subset, max_workers=max_workers, progress_callback=_cb)
        progress.empty()
        st.session_state["scan_data"] = results
        st.session_state["scan_attempted"] = list(subset.keys())

        scanned_tickers = {r["Stock"].split("(")[-1].rstrip(")") for r in results}
        failed = [t for t in subset.keys() if t not in scanned_tickers]
        if failed:
            st.warning(
                f"⚠️ {len(failed)} of {len(subset)} tickers returned no data and were skipped "
                f"(insufficient price history or a temporary Yahoo Finance issue): "
                f"{', '.join(failed)}"
            )

        if not results:
            st.warning("No results returned. Data source may be rate-limiting, or selected tickers had insufficient history.")

    data = st.session_state["scan_data"]
    if not data:
        st.info("👈 Set your scan parameters in the sidebar and click **Run Full Analysis** to begin.")
        return

    df = pd.DataFrame(data)

    # ---------------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------------
    st.subheader("📊 Screening Results")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        sector_filter = st.multiselect("Sector", options=sorted(df["Sector"].unique().tolist()), default=[])
    with fcol2:
        rating_filter = st.multiselect("Fundamental Rating", options=sorted(df["Fundamental Rating"].unique().tolist()), default=[])
    with fcol3:
        action_filter = st.multiselect("Action", options=sorted(df["Action"].unique().tolist()), default=[])
    with fcol4:
        sort_col = st.selectbox("Sort by", options=["Overall Score", "Fundamental Score", "LT Score", "ST Score", "Stock"], index=0)

    filtered = df.copy()
    if sector_filter:
        filtered = filtered[filtered["Sector"].isin(sector_filter)]
    if rating_filter:
        filtered = filtered[filtered["Fundamental Rating"].isin(rating_filter)]
    if action_filter:
        filtered = filtered[filtered["Action"].isin(action_filter)]
    filtered = filtered.sort_values(by=sort_col, ascending=False)

    st.caption(f"Showing {len(filtered)} of {len(df)} scanned stocks")

    primary_cols = ["Stock", "Sector", "Overall Score", "Combined Strategy", "Action",
                     "Fundamental Score", "Fundamental Rating", "Current Price"]
    secondary_cols = [c for c in filtered.columns if c not in primary_cols]
    display_df = filtered[primary_cols + secondary_cols]

    styler = display_df.style
    try:
        styler = styler.map(highlight_recommendations, subset=["Combined Strategy", "Action"]) \
                        .map(highlight_fundamental_rating, subset=["Fundamental Rating"])
    except AttributeError:
        # Older pandas versions (<2.1) only have applymap, not map
        styler = styler.applymap(highlight_recommendations, subset=["Combined Strategy", "Action"]) \
                        .applymap(highlight_fundamental_rating, subset=["Fundamental Rating"])
    st.dataframe(styler, use_container_width=True, height=600, hide_index=True)

    st.divider()

    # ---------------------------------------------------------------
    # LEGEND
    # ---------------------------------------------------------------
    st.subheader("📌 How to Read the Verdicts")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.success("🏆 **QUALITY BUY** — strong fundamentals *and* strong price momentum")
        st.info("💎 **ACCUMULATE** — strong fundamentals, price currently weak/consolidating")
    with lcol2:
        st.success("✅ **HOLD / ADD ON DIPS** — solid fundamentals, momentum neutral")
        st.warning("⚠️ **SPECULATIVE** — price momentum without fundamental support")
    with lcol3:
        st.error("🔴 **AVOID** — weak fundamentals, regardless of price action")
        st.caption("Fundamental Rating (Excellent/Good/Average/Weak/Poor) reflects ROE, "
                    "margins, growth, leverage and valuation. Debt/equity and current ratio "
                    "are skipped for banks/NBFCs/insurers since high leverage is structural "
                    "to their business model.")

    st.divider()

    # ---------------------------------------------------------------
    # SUMMARY METRICS
    # ---------------------------------------------------------------
    st.subheader("📈 Portfolio-Level Summary")
    quality_buy = df[df["Combined Strategy"].str.contains("QUALITY BUY", na=False)]
    accumulate = df[df["Combined Strategy"].str.contains("ACCUMULATE", na=False)]
    speculative = df[df["Combined Strategy"].str.contains("SPECULATIVE", na=False)]
    avoid = df[df["Combined Strategy"].str.contains("AVOID", na=False)]
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1: st.metric("🏆 Quality Buy", len(quality_buy))
    with mcol2: st.metric("💎 Accumulate", len(accumulate))
    with mcol3: st.metric("⚠️ Speculative", len(speculative))
    with mcol4: st.metric("🔴 Avoid", len(avoid))

    st.divider()

    # ---------------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------------
    if OPENPYXL_AVAILABLE:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            display_df.to_excel(writer, index=False, sheet_name="Nifty50 Screen")
        st.download_button(
            "⬇️ Export to Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"nifty50_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            use_container_width=True
        )
    else:
        st.download_button(
            "⬇️ Export to CSV",
            data=display_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"nifty50_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )

    st.caption(
        "💡 Tip: 'Accumulate' picks are the ones worth a closer look for long-term "
        "positions — they're businesses the model rates highly on fundamentals even "
        "when the chart looks unexciting right now. Always cross-check with a recent "
        "annual report or a source like Screener.in before acting."
    )


if __name__ == "__main__":
    main()