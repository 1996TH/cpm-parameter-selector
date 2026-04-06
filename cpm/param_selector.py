"""Automatic parameter selection for CPM.

Five methods for selecting optimal (P, T):
    1. saturation   -- Two-stage saturation (conservative)
    2. constrained  -- NE upper bound + min points
    3. curvature    -- Max curvature on Pareto front
    4. knee         -- Pareto Knee, linear normalization (default)
    5. knee_log     -- Pareto Knee, log normalization

References:
    - Bao (2008), Table 1: grid search over P and T for IBM
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cpm.config import GridSearchConfig
from cpm.cpm_core import compute_normalized_error, run_cpm


def grid_search(
    prices: np.ndarray,
    config: GridSearchConfig | None = None,
) -> pd.DataFrame:
    """Run CPM with all (P, T) combinations and compute metrics.

    Returns:
        DataFrame with columns: T, P, error_pct, error_raw, n_points, cell
        error_raw is full precision for optimization; error_pct is rounded for display.
    """
    if config is None:
        config = GridSearchConfig()

    results = []
    for T in config.T_values:
        for P in config.P_values:
            turning_points = run_cpm(prices, P, T)
            n_points = len(turning_points)
            error = compute_normalized_error(prices, turning_points)
            results.append({
                "T": T,
                "P": P,
                "error_raw": error,
                "error_pct": round(error, 1),
                "n_points": n_points,
                "cell": f"{error:.1f}%/{n_points}",
            })
    return pd.DataFrame(results)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot into Bao (2008) Table 1 format. Rows=T, Columns=P."""
    pivot = df.pivot(index="T", columns="P", values="cell")
    pivot.columns = [f"{p:.2f}" for p in pivot.columns]
    return pivot


def print_table(prices: np.ndarray, ticker: str = "", config: GridSearchConfig | None = None):
    """Run grid search and print formatted table."""
    from cpm.cpm_core import extract_local_extrema

    extrema = extract_local_extrema(prices)
    df = grid_search(prices, config)
    table = format_table(df)

    header = "Normalized error / # critical points"
    if ticker:
        header += f" -- {ticker}"
    header += f"\nOriginal data: {len(prices)} days, {len(extrema)} maximal/minimal points"
    print(header)
    print(table.to_string())
    print()
    return df


# ---------------------------------------------------------------------------
# Pareto front utilities
# ---------------------------------------------------------------------------

def _pareto_front(
    points: np.ndarray, errors: np.ndarray
) -> tuple:
    """Non-dominated set where both low error AND low points are good."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j] <= points[i] and errors[j] <= errors[i] and
                    (points[j] < points[i] or errors[j] < errors[i])):
                dominated[i] = True
                break
    idx = np.where(~dominated)[0]
    order = np.argsort(points[idx])
    return points[idx][order], errors[idx][order], idx[order]


def _find_knee(pts: np.ndarray, errs: np.ndarray) -> int:
    """Knee via normalized max perpendicular distance (linear scaling)."""
    if len(pts) < 3:
        return 0
    x = (pts - pts.min()) / (pts.max() - pts.min() + 1e-10)
    y = (errs - errs.min()) / (errs.max() - errs.min() + 1e-10)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return 0
    line_unit = line / line_len
    dists = np.array([
        abs((x[i] - p1[0]) * line_unit[1] - (y[i] - p1[1]) * line_unit[0])
        for i in range(len(x))
    ])
    return int(np.argmax(dists))


def _find_knee_log(pts: np.ndarray, errs: np.ndarray) -> int:
    """Knee via max perpendicular distance (log scaling)."""
    if len(pts) < 3:
        return 0
    x = np.log1p(pts)
    y = np.log1p(errs)
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y = (y - y.min()) / (y.max() - y.min() + 1e-10)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return 0
    line_unit = line / line_len
    dists = np.array([
        abs((x[i] - p1[0]) * line_unit[1] - (y[i] - p1[1]) * line_unit[0])
        for i in range(len(x))
    ])
    return int(np.argmax(dists))


def _find_max_curvature(pts: np.ndarray, errs: np.ndarray) -> int:
    """Maximum curvature (2nd derivative) on Pareto front."""
    if len(pts) < 5:
        return len(pts) // 2
    x = (pts - pts.min()) / (pts.max() - pts.min() + 1e-10)
    y = (errs - errs.min()) / (errs.max() - errs.min() + 1e-10)
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10) ** 1.5
    curvature[0] = 0
    curvature[-1] = 0
    return int(np.argmax(curvature))


# ---------------------------------------------------------------------------
# Selection methods (all use error_raw for precision)
# ---------------------------------------------------------------------------

def _select_knee(df: pd.DataFrame) -> tuple:
    """Method 4: Pareto Knee (Linear normalization)."""
    pts = df["n_points"].values.astype(float)
    errs = df["error_raw"].values.astype(float)
    fp, fe, fi = _pareto_front(pts, errs)
    k = _find_knee(fp, fe)
    return df.iloc[fi[k]]["P"], int(df.iloc[fi[k]]["T"])


def _select_saturation(df: pd.DataFrame, threshold: float = 0.02) -> tuple:
    """Method 1: Two-stage saturation."""
    total_pts = df["n_points"].max()

    sat_per_T = []
    for T in sorted(df["T"].unique()):
        rows = df[df["T"] == T].sort_values("P")
        Ps = rows["P"].values
        pts = rows["n_points"].values
        errs = rows["error_raw"].values

        sat_P, sat_err, sat_pts = Ps[0], errs[0], pts[0]
        for i in range(1, len(Ps)):
            if (pts[i - 1] - pts[i]) / total_pts < threshold:
                sat_P, sat_err, sat_pts = Ps[i - 1], errs[i - 1], pts[i - 1]
                break
        else:
            sat_P, sat_err, sat_pts = Ps[-1], errs[-1], pts[-1]
        sat_per_T.append({"T": T, "P": sat_P, "error": sat_err, "points": sat_pts})

    for i in range(1, len(sat_per_T)):
        reduction = (sat_per_T[i - 1]["points"] - sat_per_T[i]["points"]) / total_pts
        if reduction < threshold:
            return sat_per_T[i - 1]["P"], int(sat_per_T[i - 1]["T"])

    return sat_per_T[-1]["P"], int(sat_per_T[-1]["T"])


def _select_constrained(df: pd.DataFrame, max_error=None) -> tuple:
    """Method 2: Within NE <= max_error, find min # critical points."""
    if max_error is None:
        max_error = df["error_raw"].median()
    valid = df[df["error_raw"] <= max_error]
    if len(valid) == 0:
        valid = df.nsmallest(1, "error_raw")
    best = valid.loc[valid["n_points"].idxmin()]
    return best["P"], int(best["T"])


def _select_curvature(df: pd.DataFrame) -> tuple:
    """Method 3: Maximum curvature on Pareto front."""
    pts = df["n_points"].values.astype(float)
    errs = df["error_raw"].values.astype(float)
    fp, fe, fi = _pareto_front(pts, errs)
    k = _find_max_curvature(fp, fe)
    return df.iloc[fi[k]]["P"], int(df.iloc[fi[k]]["T"])


def _select_knee_log(df: pd.DataFrame) -> tuple:
    """Method 5: Pareto Knee (Log normalization)."""
    pts = df["n_points"].values.astype(float)
    errs = df["error_raw"].values.astype(float)
    fp, fe, fi = _pareto_front(pts, errs)
    k = _find_knee_log(fp, fe)
    return df.iloc[fi[k]]["P"], int(df.iloc[fi[k]]["T"])


METHODS = {
    "saturation": _select_saturation,   # 1
    "constrained": _select_constrained, # 2
    "curvature": _select_curvature,     # 3
    "knee": _select_knee,               # 4
    "knee_log": _select_knee_log,       # 5
}


def auto_select(
    prices: np.ndarray,
    config: GridSearchConfig | None = None,
    method: str = "knee",
) -> tuple:
    """Automatically select optimal (P, T).

    Args:
        prices: 1-D close price array.
        config: Grid search ranges.
        method: "knee" (default), "saturation", "constrained", "curvature", "knee_log"

    Returns:
        (P, T) tuple.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS)}")
    df = grid_search(prices, config)
    return METHODS[method](df)
