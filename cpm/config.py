"""Grid search configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GridSearchConfig:
    """P, T grid search ranges (Bao 2008 Table 1 style)."""

    P_values: list = field(
        default_factory=lambda: [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    )
    T_values: list = field(
        default_factory=lambda: [4, 6, 8, 10, 12, 14, 16]
    )


def grid_config_from_ranges(
    p_min: float,
    p_max: float,
    p_step: float,
    t_min: int,
    t_max: int,
    t_step: int,
) -> GridSearchConfig:
    """Build ``GridSearchConfig`` from inclusive min/max and step (Bao-style grid)."""
    if p_step <= 0:
        raise ValueError("p_step must be positive")
    if t_step <= 0:
        raise ValueError("t_step must be positive")
    if p_min > p_max:
        raise ValueError("p_min must be <= p_max")
    if t_min > t_max:
        raise ValueError("t_min must be <= t_max")

    p_values: list[float] = []
    x = p_min
    while x <= p_max + 1e-9:
        p_values.append(round(float(x), 10))
        x += p_step

    t_values = list(range(int(t_min), int(t_max) + 1, int(t_step)))

    if not p_values:
        raise ValueError("P grid is empty; check p_min, p_max, p_step")
    if not t_values:
        raise ValueError("T grid is empty; check t_min, t_max, t_step")

    return GridSearchConfig(P_values=p_values, T_values=t_values)


def default_cli_grid_bounds() -> tuple[float, float, float, int, int, int]:
    """(p_min, p_max, p_step, t_min, t_max, t_step) matching :class:`GridSearchConfig` defaults."""
    c = GridSearchConfig()
    pv, tv = c.P_values, c.T_values
    if len(pv) < 1 or len(tv) < 1:
        raise RuntimeError("GridSearchConfig default lists must be non-empty")
    p_step = float(pv[1] - pv[0]) if len(pv) > 1 else 0.02
    t_step = int(tv[1] - tv[0]) if len(tv) > 1 else 1
    return float(pv[0]), float(pv[-1]), p_step, int(tv[0]), int(tv[-1]), t_step
