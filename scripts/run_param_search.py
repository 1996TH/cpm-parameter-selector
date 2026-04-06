"""Run P,T grid search + auto selection for specified assets.

Usage (after pip install -e .):
    python -m scripts.run_param_search
    python -m scripts.run_param_search SPY GLD
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpm import load_prices, auto_select, print_table

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO", "BITO"]


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"  {ticker}")
        print(f"{'='*50}")

        prices = load_prices(ticker)
        print(f"  {len(prices)} days loaded\n")

        print_table(prices, ticker=ticker)

        print("  Auto-selection results:")
        for method in ["knee", "saturation"]:
            P, T = auto_select(prices, method=method)
            print(f"    {method:14s} -> P={P:.2f}, T={T}")
        print()


if __name__ == "__main__":
    main()
