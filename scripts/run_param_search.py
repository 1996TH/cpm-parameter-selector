"""Run P,T grid search + auto selection + visualization for specified assets.

Usage (after python -m pip install -e .):
    python scripts/run_param_search.py
    python scripts/run_param_search.py SPY GLD
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from cpm import (
    auto_select,
    extract_local_extrema,
    grid_search,
    load_prices,
    print_table,
    render_table_image,
    run_cpm,
    to_triangle_wave,
    compute_normalized_error,
)
from cpm.param_selector import _pareto_front, _find_knee

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO", "BITO"]


def plot_cpm(prices, ticker, P, T, save_dir):
    """Price + turning points + triangle wave."""
    tp = run_cpm(prices, P, T)
    wave = to_triangle_wave(tp, len(prices))
    err = compute_normalized_error(prices, tp)
    days = np.arange(len(prices))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(days, prices, color="gray", linewidth=0.6, alpha=0.5)
    tp_x = [p.x for p in tp]
    tp_y = [p.y for p in tp]
    ax1.plot(tp_x, tp_y, color="#3498db", linewidth=1.5)
    for p in tp:
        color = "#e74c3c" if p.is_max else "#2ecc71"
        marker = "v" if p.is_max else "^"
        ax1.plot(p.x, p.y, marker, color=color, markersize=7)
    ax1.set_title(f"{ticker} -- P={P}, T={T}  |  {len(tp)} points  |  error={err:.1f}%",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price")

    ax2.plot(days, wave, color="#8e44ad", linewidth=1.0)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_ylabel("Signal")
    ax2.set_xlabel("Days")
    ax2.fill_between(days, wave, 0, where=(wave < 0), alpha=0.15, color="#2ecc71")
    ax2.fill_between(days, wave, 0, where=(wave > 0), alpha=0.15, color="#e74c3c")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{ticker}_P{P}_T{T}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def plot_pareto(prices, ticker, save_dir):
    """Pareto front with Saturation and Knee points."""
    from cpm.param_selector import _select_saturation, _select_knee

    df = grid_search(prices)
    pts = df["n_points"].values.astype(float)
    errs = df["error_raw"].values.astype(float)
    labels = [f"P={r['P']:.2f}, T={int(r['T'])}" for _, r in df.iterrows()]

    fp, fe, fi = _pareto_front(pts, errs)
    k = _find_knee(fp, fe)
    knee_idx = fi[k]

    sat_P, sat_T = _select_saturation(df)
    sat_row = df[(df["P"] == sat_P) & (df["T"] == sat_T)].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(pts, errs, color="#dddddd", s=20, zorder=1)
    ax.plot(fp, fe, "o-", color="#3498db", linewidth=2, markersize=5, zorder=2, label="Pareto front")

    # Knee
    ax.plot(pts[knee_idx], errs[knee_idx], "*", color="#e74c3c", markersize=25, zorder=4,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate(labels[knee_idx], (pts[knee_idx], errs[knee_idx]),
               textcoords="offset points", xytext=(12, 12), fontsize=10, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", edgecolor="#e74c3c"),
               arrowprops=dict(arrowstyle="->", color="#e74c3c", linewidth=1.5))

    # Saturation
    ax.plot(sat_row["n_points"], sat_row["error_raw"], "s", color="#27ae60", markersize=14, zorder=4,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate(f"P={sat_P:.2f}, T={sat_T}", (sat_row["n_points"], sat_row["error_raw"]),
               textcoords="offset points", xytext=(12, -15), fontsize=10, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccffcc", edgecolor="#27ae60"),
               arrowprops=dict(arrowstyle="->", color="#27ae60", linewidth=1.5))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#3498db", linewidth=2, markersize=5, label="Pareto front"),
        Line2D([0], [0], marker="*", color="#e74c3c", markersize=15, linestyle="None", label="Pareto Knee"),
        Line2D([0], [0], marker="s", color="#27ae60", markersize=10, linestyle="None", label="Saturation"),
    ]
    ax.legend(handles=handles, fontsize=11)
    ax.set_xlabel("# Critical Points", fontsize=13)
    ax.set_ylabel("Normalized Error (%)", fontsize=13)
    ax.set_title(f"{ticker} -- Pareto Front", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{ticker}_pareto.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def run_asset(ticker):
    """Full pipeline for one asset: table + auto-select + charts."""
    save_dir = os.path.join("output", ticker)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {ticker}")
    print(f"{'='*50}")

    prices = load_prices(ticker)
    extrema = extract_local_extrema(prices)
    print(f"  {len(prices)} days, {len(extrema)} extrema\n")

    # 1. Grid search table (console + image)
    df = print_table(prices, ticker=ticker)
    render_table_image(df, ticker, os.path.join(save_dir, "grid_search_table.png"))
    print(f"  Saved: {save_dir}/grid_search_table.png")

    # 2. Auto-selection
    results = {}
    for method in ["knee", "saturation"]:
        P, T = auto_select(prices, method=method)
        results[method] = (P, T)
        print(f"  {method:14s} -> P={P:.2f}, T={T}")

    # 3. CPM + triangle wave for each method
    for method, (P, T) in results.items():
        fname = plot_cpm(prices, ticker, P, T, save_dir)
        print(f"  Saved: {fname}")

    # 4. Pareto front
    fname = plot_pareto(prices, ticker, save_dir)
    print(f"  Saved: {fname}")

    print(f"\n  Done: {save_dir}/")


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    for ticker in tickers:
        run_asset(ticker)


if __name__ == "__main__":
    main()
