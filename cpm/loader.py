"""Load close prices from datakit CSV, fallback to yfinance."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _get_ohlcv_dir() -> Path:
    return Path(
        os.environ.get(
            "DATAKIT_OHLCV_DIR",
            Path(__file__).resolve().parent.parent.parent / "datakit" / "data" / "ohlcv",
        )
    )


_PERIOD_OFFSETS = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "5y": pd.DateOffset(years=5),
    "10y": pd.DateOffset(years=10),
}


def _filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "max" or df.empty:
        return df
    dates = pd.to_datetime(df["Date"], format="mixed", utc=True)
    latest = dates.max()
    if period == "ytd":
        start = pd.Timestamp(year=latest.year, month=1, day=1, tz="UTC")
    elif period in _PERIOD_OFFSETS:
        start = latest - _PERIOD_OFFSETS[period]
    else:
        warnings.warn(f"Unknown period '{period}', returning all data")
        return df
    return df.loc[dates >= start]


def load_prices(ticker: str, period: str = "max") -> np.ndarray:
    csv_path = _get_ohlcv_dir() / f"{ticker}.csv"

    if not csv_path.exists():
        warnings.warn(f"{ticker}: CSV not found at {csv_path}, falling back to yfinance")
        return _fallback_yfinance(ticker, period)

    df = pd.read_csv(csv_path)

    if df.empty:
        warnings.warn(f"{ticker}: CSV is empty, falling back to yfinance")
        return _fallback_yfinance(ticker, period)

    if "Close" not in df.columns:
        warnings.warn(f"{ticker}: CSV missing 'Close' column, falling back to yfinance")
        return _fallback_yfinance(ticker, period)

    df = _filter_by_period(df, period)
    prices = df["Close"].to_numpy(dtype=float).flatten()

    if len(prices) == 0:
        warnings.warn(f"{ticker}: no data after period='{period}' filter, falling back to yfinance")
        return _fallback_yfinance(ticker, period)

    return prices


def _fallback_yfinance(ticker: str, period: str) -> np.ndarray:
    import yfinance as yf

    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    prices = df["Close"].values.flatten()
    if len(prices) == 0:
        raise ValueError(f"Empty price series for ticker '{ticker}'")
    return prices
