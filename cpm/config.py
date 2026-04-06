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
