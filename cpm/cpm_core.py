"""Critical Point Model (CPM) core algorithm.

Implements Bao (2008) Section 2.3 three-point unit algorithm for
robust turning point detection in financial time series.

References:
    - Bao (2008), "A generalized model for financial time series
      representation and prediction"
    - Bao & Yang (2008), "Intelligent stock trading system by turning
      point confirming and probabilistic reasoning"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CriticalPoint:
    """A critical point in the price series."""

    x: int  # time index
    y: float  # price value
    is_max: bool  # True = local max, False = local min


def extract_local_extrema(prices: np.ndarray) -> list[CriticalPoint]:
    """Extract all local maxima and minima, alternating.

    Handles plateaus by taking the midpoint of flat regions.
    Ensures max and min alternate strictly.
    """
    n = len(prices)
    if n < 3:
        return [CriticalPoint(0, prices[0], False), CriticalPoint(n - 1, prices[-1], True)]

    extrema: list[CriticalPoint] = []

    i = 1
    while i < n - 1:
        j = i
        while j < n - 1 and prices[j] == prices[i]:
            j += 1

        mid = (i + j - 1) // 2

        left = prices[i - 1]
        right = prices[j] if j < n else prices[j - 1]
        val = prices[mid]

        if val > left and val > right:
            extrema.append(CriticalPoint(mid, val, is_max=True))
        elif val < left and val < right:
            extrema.append(CriticalPoint(mid, val, is_max=False))

        i = j if j > i else i + 1

    if len(extrema) < 2:
        return extrema

    filtered = [extrema[0]]
    for k in range(1, len(extrema)):
        if extrema[k].is_max == filtered[-1].is_max:
            if extrema[k].is_max:
                if extrema[k].y > filtered[-1].y:
                    filtered[-1] = extrema[k]
            else:
                if extrema[k].y < filtered[-1].y:
                    filtered[-1] = extrema[k]
        else:
            filtered.append(extrema[k])

    return filtered


def exceeds_threshold(a: CriticalPoint, b: CriticalPoint, P: float, T: int) -> bool:
    """Check if oscillation between two points exceeds P or duration exceeds T."""
    mean_price = (a.y + b.y) / 2.0
    if mean_price == 0:
        return True
    oscillation = abs(a.y - b.y) / mean_price
    duration = abs(b.x - a.x)

    return oscillation >= P or duration >= T


def run_cpm(prices: np.ndarray, P: float, T: int) -> list[CriticalPoint]:
    """Run the Critical Point Model (3-point unit algorithm).

    Bao (2008) Section 2.3: iteratively processes 3-point units,
    classifying into Cases 1-4 and filtering noise.

    Args:
        prices: 1-D close price array.
        P: price oscillation threshold (e.g. 0.06 = 6%).
        T: time duration threshold in days.

    Returns:
        List of preserved CriticalPoints (turning points).
    """
    C = extract_local_extrema(prices)

    if len(C) < 3:
        return C

    first_min_idx = 0
    for idx, cp in enumerate(C):
        if not cp.is_max:
            first_min_idx = idx
            break

    if first_min_idx + 2 >= len(C):
        return C

    preserved: list[CriticalPoint] = []

    for idx in range(first_min_idx):
        preserved.append(C[idx])

    i = first_min_idx
    i1 = first_min_idx + 1
    i2 = first_min_idx + 2

    while i2 < len(C):
        left_exceeds = exceeds_threshold(C[i], C[i1], P, T)
        right_exceeds = exceeds_threshold(C[i1], C[i2], P, T)

        if left_exceeds and right_exceeds:
            preserved.append(C[i])
            preserved.append(C[i1])
            i = i2
            i1 = i2 + 1
            i2 = i2 + 2

        elif left_exceeds and not right_exceeds:
            i3 = i2 + 1
            if i3 >= len(C):
                preserved.append(C[i])
                preserved.append(C[i1])
                i = i2
                i1 = i2 + 1
                i2 = i2 + 2
                break

            if C[i1].is_max:
                if C[i3].y >= C[i1].y:
                    i1 = i3
                    i2 = i3 + 1
                else:
                    i2 = i3 + 1
            else:
                if C[i3].y <= C[i1].y:
                    i1 = i3
                    i2 = i3 + 1
                else:
                    i2 = i3 + 1

        elif not left_exceeds and right_exceeds:
            i = i2
            i1 = i2 + 1
            i2 = i2 + 2

        else:
            i3 = i2 + 1
            i4 = i2 + 2

            if C[i].is_max:
                keep = i if C[i].y >= C[i2].y else i2
            else:
                keep = i if C[i].y <= C[i2].y else i2

            if i4 < len(C):
                i = keep
                i1 = i3
                i2 = i4
            else:
                preserved.append(C[i])
                preserved.append(C[i1])
                i = i2
                break

    if not preserved or preserved[-1].x != C[-1].x:
        preserved.append(C[-1])

    if preserved[-1].x != C[min(i, len(C) - 1)].x:
        preserved.insert(-1, C[min(i, len(C) - 1)])

    return preserved


def compute_normalized_error(prices: np.ndarray, turning_points: list[CriticalPoint]) -> float:
    """Compute approximation error between original prices and piecewise linear fit.

    Returns:
        Normalized error as percentage (e.g. 3.5 means 3.5%).
    """
    if len(turning_points) < 2:
        return 100.0

    n = len(prices)
    approx = np.full(n, np.nan)

    for k in range(len(turning_points) - 1):
        start = turning_points[k]
        end = turning_points[k + 1]
        if start.x == end.x:
            approx[start.x] = start.y
            continue
        for t in range(start.x, min(end.x + 1, n)):
            ratio = (t - start.x) / (end.x - start.x)
            approx[t] = start.y + ratio * (end.y - start.y)

    first_x = turning_points[0].x
    last_x = turning_points[-1].x
    for t in range(0, first_x):
        approx[t] = prices[t]
    for t in range(last_x + 1, n):
        approx[t] = prices[t]

    valid = ~np.isnan(approx)
    if not np.any(valid):
        return 100.0

    mean_price = np.mean(prices[valid])
    if mean_price == 0:
        return 100.0

    error = np.mean(np.abs(prices[valid] - approx[valid])) / mean_price * 100.0
    return error


def to_triangle_wave(
    turning_points: list[CriticalPoint], total_days: int
) -> np.ndarray:
    """Convert turning points to triangle wave target signal [-1, +1].

    Local min -> -1, local max -> +1, linear interpolation between.
    """
    target = np.zeros(total_days)

    if len(turning_points) < 2:
        return target

    for k in range(len(turning_points) - 1):
        start = turning_points[k]
        end = turning_points[k + 1]

        start_val = -1.0 if not start.is_max else 1.0
        end_val = -1.0 if not end.is_max else 1.0

        if start.x == end.x:
            target[start.x] = start_val
            continue

        for t in range(start.x, min(end.x + 1, total_days)):
            ratio = (t - start.x) / (end.x - start.x)
            target[t] = start_val + ratio * (end_val - start_val)

    if turning_points[0].x > 0:
        target[: turning_points[0].x] = -1.0 if not turning_points[0].is_max else 1.0
    if turning_points[-1].x < total_days - 1:
        target[turning_points[-1].x :] = -1.0 if not turning_points[-1].is_max else 1.0

    return target
