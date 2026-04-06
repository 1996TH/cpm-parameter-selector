"""Price data loading utilities."""

from __future__ import annotations

import numpy as np


def load_prices(ticker: str, period: str = "max") -> np.ndarray:
    """Load close prices from yfinance.

    Args:
        ticker: ETF/stock ticker symbol.
        period: yfinance period string.

    Returns:
        1-D numpy array of close prices.

    Raises:
        ValueError: If no data is returned for the ticker.
    """
    import yfinance as yf

    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    prices = df["Close"].values.flatten()
    if len(prices) == 0:
        raise ValueError(f"Empty price series for ticker '{ticker}'")
    return prices
