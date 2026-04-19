"""Run P,T grid search + auto selection + visualization for specified assets.

Usage (after ``python -m pip install -e .`` from the project root):

  기본 그리드·티커
    ``GridSearchConfig`` 와 동일한 P/T 범위·간격(기본값)과, 내장 ETF 목록(SPY, QQQ, …)으로 실행.

    python scripts/run_param_search.py

  티커만 지정
    그리드는 기본값 그대로, 심볼만 바꿔서 실행.

    python scripts/run_param_search.py SPY GLD

  메소드·P/T 그리드·티커 지정
    - ``--method``: 자동 선택 방식 (knee, saturation, constrained, curvature, knee_log)
    - ``--p-min`` / ``--p-max`` / ``--p-step``: 가격 변동 임계 P의 탐색 구간·간격
    - ``--t-min`` / ``--t-max`` / ``--t-step``: 전환점 최소 간격 T(일)의 구간·간격
    아래는 기본 그리드와 동일한 값의 예시이며, 이 플래그들을 모두 생략해도 같은 기본 그리드가 쓰임.

    python scripts/run_param_search.py --method knee --p-min 0.02 --p-max 0.18 --p-step 0.02 --t-min 4 --t-max 16 --t-step 2 SPY
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from cpm import (
    METHODS,
    auto_select,
    extract_local_extrema,
    load_prices,
    print_table,
    render_table_image,
    run_cpm,
    to_triangle_wave,
    compute_normalized_error,
)
from cpm.config import default_cli_grid_bounds, grid_config_from_ranges
from cpm.param_selector import _pareto_front

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO", "BITO"]
METHOD_CHOICES = tuple(sorted(METHODS.keys()))


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


def plot_pareto(ticker, save_dir, df, method: str, P_sel: float, T_sel: int):
    """Pareto front with the selected method's (P, T) highlighted."""
    pts = df["n_points"].values.astype(float)
    errs = df["error_raw"].values.astype(float)

    fp, fe, _ = _pareto_front(pts, errs)

    sel = df[
        np.isclose(df["P"].to_numpy(dtype=float), float(P_sel))
        & (df["T"].to_numpy() == int(T_sel))
    ]
    if len(sel) == 0:
        sel = df.iloc[[0]]
    row = sel.iloc[0]
    sel_label = f"{method}: P={row['P']:.2f}, T={int(row['T'])}"

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(pts, errs, color="#dddddd", s=20, zorder=1)
    ax.plot(fp, fe, "o-", color="#3498db", linewidth=2, markersize=5, zorder=2, label="Pareto front")

    ax.plot(row["n_points"], row["error_raw"], "*", color="#e74c3c", markersize=25, zorder=4,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate(
        sel_label,
        (row["n_points"], row["error_raw"]),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeecc", edgecolor="#e67e22"),
        arrowprops=dict(arrowstyle="->", color="#e67e22", linewidth=1.5),
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#3498db", linewidth=2, markersize=5, label="Pareto front"),
        Line2D([0], [0], marker="*", color="#e74c3c", markersize=15, linestyle="None", label=f"Selected ({method})"),
    ]
    ax.legend(handles=handles, fontsize=11)
    ax.set_xlabel("# Critical Points", fontsize=13)
    ax.set_ylabel("Normalized Error (%)", fontsize=13)
    ax.set_title(f"{ticker} -- Pareto Front ({method})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{ticker}_pareto.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def run_asset(ticker, method: str, config):
    """Full pipeline for one asset: table + auto-select + charts."""
    save_dir = os.path.join("output", ticker)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- {ticker} ---")
    pv, tv = config.P_values, config.T_values
    p_part = f"P in [{pv[0]:.4g}..{pv[-1]:.4g}]"
    if len(pv) > 1:
        p_part += f" step {pv[1] - pv[0]:.4g}"
    t_part = f"T in [{tv[0]}..{tv[-1]}]"
    if len(tv) > 1:
        t_part += f" step {tv[1] - tv[0]}"
    print(f"  method={method}, {p_part}")
    print(f"  {t_part}")

    prices = load_prices(ticker)
    extrema = extract_local_extrema(prices)
    print(f"  {len(prices)} days, {len(extrema)} extrema\n")

    # 1. Grid search table (console + image)
    df = print_table(prices, ticker=ticker, config=config)
    render_table_image(df, ticker, os.path.join(save_dir, "grid_search_table.png"))
    print(f"  Saved: {save_dir}/grid_search_table.png")

    # 2. Auto-selection
    P, T = auto_select(prices, config=config, method=method)
    print(f"  {method:14s} -> P={P:.2f}, T={T}")

    # 3. CPM + triangle wave
    fname = plot_cpm(prices, ticker, P, T, save_dir)
    print(f"  Saved: {fname}")

    # 4. Pareto front
    fname = plot_pareto(ticker, save_dir, df, method, P, T)
    print(f"  Saved: {fname}")

    print(f"\n  Done: {save_dir}/")


def _build_arg_parser() -> argparse.ArgumentParser:
    dp_min, dp_max, dp_step, dt_min, dt_max, dt_step = default_cli_grid_bounds()
    p = argparse.ArgumentParser(
        description="CPM grid search and automatic (P, T) selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "tickers",
        nargs="*",
        help="Symbols to process (default: built-in ETF list)",
    )
    p.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="knee",
        help="Selection method",
    )
    p.add_argument(
        "--p-min",
        type=float,
        default=dp_min,
        help="P grid minimum (defaults match cpm.config.GridSearchConfig)",
    )
    p.add_argument("--p-max", type=float, default=dp_max, help="P grid maximum (inclusive)")
    p.add_argument("--p-step", type=float, default=dp_step, help="P grid step")
    p.add_argument("--t-min", type=int, default=dt_min, help="T grid minimum (days)")
    p.add_argument("--t-max", type=int, default=dt_max, help="T grid maximum (days, inclusive)")
    p.add_argument("--t-step", type=int, default=dt_step, help="T grid step (days)")
    return p


def main():
    args = _build_arg_parser().parse_args()
    try:
        config = grid_config_from_ranges(
            args.p_min,
            args.p_max,
            args.p_step,
            args.t_min,
            args.t_max,
            args.t_step,
        )
    except ValueError as e:
        print(f"Invalid grid: {e}", file=sys.stderr)
        sys.exit(2)

    tickers = args.tickers if args.tickers else TICKERS
    for ticker in tickers:
        run_asset(ticker, args.method, config)


if __name__ == "__main__":
    main()
