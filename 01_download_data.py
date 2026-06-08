"""
STAGE 1 -- Data acquisition & dataset construction.

Downloads daily adjusted-close prices for the 15-stock Indian portfolio from
Yahoo Finance, builds the market-cap weights (=> rupee allocation of Rs 1 cr),
and saves four clearly-labelled CSV datasets into ./data/ :

    prices_adjusted_close.csv  -- split/dividend-adjusted close, one column/stock
    daily_returns.csv          -- asset-level simple daily returns
    portfolio_weights.csv      -- market cap, weight, rupee allocation per stock
    portfolio_returns.csv      -- single weighted portfolio return series (+ Rs P&L)

Run:  python 01_download_data.py
"""

import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf

import config as C


def download_prices() -> pd.DataFrame:
    """Adjusted close for all tickers, 2018 -> download_end."""
    print(f"Downloading {len(C.TICKERS)} tickers "
          f"{C.HISTORY_START} -> {C.DOWNLOAD_END} ...")
    raw = yf.download(
        C.TICKERS,
        start=C.HISTORY_START,
        end=C.DOWNLOAD_END,
        auto_adjust=True,       # 'Close' becomes split/dividend adjusted
        progress=False,
        group_by="column",
        threads=True,
    )
    # With multiple tickers and group_by="column" we get a column MultiIndex
    # ('Close','RELIANCE.NS'), ... -- pull just the Close block.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:                       # single ticker edge-case
        prices = raw[["Close"]].copy()
        prices.columns = C.TICKERS

    prices = prices.reindex(columns=C.TICKERS)   # keep order
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Data quality report
    missing = prices.isna().sum()
    print("\nMissing values per ticker (pre-clean):")
    print(missing.to_string())

    # Forward-fill small gaps, then drop any leading rows still NaN
    prices = prices.ffill().dropna(how="any")
    print(f"\nClean price panel: {prices.shape[0]} trading days x "
          f"{prices.shape[1]} stocks")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    return prices


def fetch_market_caps() -> pd.Series:
    """Live market caps (INR). Falls back to config table on failure."""
    caps = {}
    print("\nFetching live market caps ...")
    for t in C.TICKERS:
        cap = None
        try:
            info = yf.Ticker(t).get_info()
            cap = info.get("marketCap", None)
        except Exception as e:
            print(f"  {t}: live fetch failed ({e.__class__.__name__})")
        if cap:
            caps[t] = float(cap)
            print(f"  {t}: Rs {cap/1e7:,.0f} cr (live)")
        else:
            caps[t] = float(C.FALLBACK_MARKET_CAP_CR[t]) * 1e7  # cr -> rupees
            print(f"  {t}: Rs {C.FALLBACK_MARKET_CAP_CR[t]:,.0f} cr (fallback)")
        time.sleep(0.2)   # be polite to the API
    return pd.Series(caps, name="market_cap_inr")


def build_weights(caps: pd.Series) -> pd.DataFrame:
    if C.WEIGHTING == "market_cap":
        w = caps / caps.sum()
    else:
        w = pd.Series(1.0 / len(C.TICKERS), index=C.TICKERS)
    df = pd.DataFrame({
        "sector": pd.Series(C.SECTOR),
        "market_cap_inr": caps,
        "weight": w,
        "rupee_allocation": (w * C.PORTFOLIO_VALUE_INR).round(2),
    }).loc[C.TICKERS]
    df.index.name = "ticker"
    return df


def main():
    prices = download_prices()
    if prices.shape[0] < 250:
        sys.exit("ERROR: too few rows downloaded -- aborting.")

    # Asset-level simple returns (asset-additive => correct for aggregation)
    returns = prices.pct_change().dropna(how="any")

    caps = fetch_market_caps()
    weights = build_weights(caps)
    w = weights["weight"].values

    # Constant-weight portfolio simple return = sum_i w_i * R_i
    port_ret = pd.Series(returns.values @ w, index=returns.index,
                         name="portfolio_return")
    port_df = pd.DataFrame({
        "portfolio_return": port_ret,
        "pnl_inr": (port_ret * C.PORTFOLIO_VALUE_INR).round(2),
    })

    # ---- save everything, clearly labelled ----
    prices.round(4).to_csv(C.PRICES_CSV)
    returns.round(6).to_csv(C.RETURNS_CSV)
    weights.to_csv(C.WEIGHTS_CSV)
    port_df.to_csv(C.PORT_RET_CSV)

    print("\nSaved datasets:")
    for p in (C.PRICES_CSV, C.RETURNS_CSV, C.WEIGHTS_CSV, C.PORT_RET_CSV):
        print(f"  {p}")

    print("\nPortfolio weights (market-cap, Rs 1 cr):")
    print(weights[["sector", "weight", "rupee_allocation"]]
          .assign(weight=lambda d: (d["weight"] * 100).round(2).astype(str) + "%")
          .to_string())
    print(f"\nTotal allocation: Rs {weights['rupee_allocation'].sum():,.0f}")
    print(f"Portfolio return series: {len(port_ret)} days, "
          f"{port_ret.index.min().date()} -> {port_ret.index.max().date()}")


if __name__ == "__main__":
    main()
