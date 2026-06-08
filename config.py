"""
Central configuration for the Indian-equity portfolio VaR/CVaR project.

Everything downstream (data download, risk models, backtest, prediction)
imports from here so there is a single source of truth.

All monetary figures are in INR (rupees).
"""

from datetime import date

# ---------------------------------------------------------------------------
# 1. PORTFOLIO UNIVERSE  --  Diversified 15 Indian large-caps (less bank-heavy)
#    Yahoo Finance NSE tickers (".NS" suffix).
# ---------------------------------------------------------------------------
TICKERS = [
    "RELIANCE.NS",    # Energy / conglomerate
    "TCS.NS",         # IT services
    "INFY.NS",        # IT services
    "HDFCBANK.NS",    # Banking
    "ICICIBANK.NS",   # Banking
    "SBIN.NS",        # Banking (PSU)
    "HINDUNILVR.NS",  # FMCG (defensive)
    "ITC.NS",         # FMCG / diversified
    "LT.NS",          # Infrastructure / engineering
    "BHARTIARTL.NS",  # Telecom
    "MARUTI.NS",      # Auto
    "M&M.NS",         # Auto
    "SUNPHARMA.NS",   # Pharma (defensive)
    "TATASTEEL.NS",   # Metals (cyclical)
    "ASIANPAINT.NS",  # Consumer / paints
]

SECTOR = {
    "RELIANCE.NS": "Energy",        "TCS.NS": "IT",            "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",       "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",        "ITC.NS": "FMCG",          "LT.NS": "Infra",
    "BHARTIARTL.NS": "Telecom",     "MARUTI.NS": "Auto",       "M&M.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",       "TATASTEEL.NS": "Metals",  "ASIANPAINT.NS": "Consumer",
}

# Fallback market caps (approx, INR crore) used ONLY if the live fetch fails.
# Relative magnitudes are what matter for cap weighting.
FALLBACK_MARKET_CAP_CR = {
    "RELIANCE.NS": 1900000, "TCS.NS": 1500000, "HDFCBANK.NS": 1300000,
    "BHARTIARTL.NS": 900000, "ICICIBANK.NS": 850000, "SBIN.NS": 720000,
    "INFY.NS": 650000, "ITC.NS": 580000, "HINDUNILVR.NS": 550000,
    "LT.NS": 500000, "MARUTI.NS": 420000, "SUNPHARMA.NS": 420000,
    "M&M.NS": 350000, "ASIANPAINT.NS": 230000, "TATASTEEL.NS": 200000,
}

# ---------------------------------------------------------------------------
# 2. PORTFOLIO ECONOMICS
# ---------------------------------------------------------------------------
PORTFOLIO_VALUE_INR = 1_00_00_000      # Rupees 1 crore
WEIGHTING = "market_cap"                # "market_cap" or "equal"

# ---------------------------------------------------------------------------
# 3. RISK PARAMETERS
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS = [0.95, 0.99]       # VaR / CVaR confidence levels
HORIZONS_DAYS = {"1-day": 1, "1-week": 5}   # 1 week = 5 trading days
MC_SIMULATIONS = 100_000               # Monte Carlo scenario count
MC_SEED = 42                           # reproducibility

# ---------------------------------------------------------------------------
# 4. DATES
#    - History starts 2018 (captures COVID-2020 crash for backtesting).
#    - Train strictly BEFORE 1 March 2026.
#    - Out-of-sample backtest: 2 March 2026 -> 6 June 2026.
#    - "Today" prediction is made from the latest available data.
# ---------------------------------------------------------------------------
HISTORY_START = "2018-01-01"
TRAIN_END = "2026-03-01"               # exclusive cut-off (train on dates < this)
BACKTEST_END = "2026-06-06"
TODAY = "2026-06-08"
DOWNLOAD_END = "2026-06-09"            # yfinance 'end' is exclusive; pad by 1 day

# ---------------------------------------------------------------------------
# 5. PATHS
# ---------------------------------------------------------------------------
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PRICES_CSV = os.path.join(DATA_DIR, "prices_adjusted_close.csv")
RETURNS_CSV = os.path.join(DATA_DIR, "daily_returns.csv")   # asset-level simple returns
WEIGHTS_CSV = os.path.join(DATA_DIR, "portfolio_weights.csv")
PORT_RET_CSV = os.path.join(DATA_DIR, "portfolio_returns.csv")
