# cpm-param-selector

Automatic (P, T) parameter selection for CPM (Critical Point Model).
Determines noise filtering level when finding turning points in price series.

- P: price oscillation threshold (e.g. 0.06 = 6%)
- T: minimum time gap between turning points (days)

## References

- Bao (2008), "A generalized model for financial time series representation and prediction"
- Bao & Yang (2008), "Intelligent stock trading system by turning point confirming and probabilistic reasoning"
- Lin et al. (2011), "Intelligent stock trading system based on improved technical analysis and Echo State Network"

Extends Bao (2008) Table 1 grid search to multi-asset ETFs + 9 auto-selection methods.

## Structure

```
cpm/
├── cpm_core.py         # CPM algorithm (extrema, 3-point unit, error)
├── loader.py           # price loading (datakit CSV -> yfinance fallback)
├── param_selector.py   # grid search + auto selection
├── config.py           # P, T range config
└── visualize.py        # table/chart rendering
scripts/
└── run_param_search.py
```

## Usage

```bash
pip install -e .

# run script
python scripts/run_param_search.py SPY QQQ GLD

# as library
from cpm import load_prices, auto_select, grid_search

prices = load_prices("SPY")
P, T = auto_select(prices, method="knee")
P, T = auto_select(prices, method="saturation")
```

## Selection methods

- `knee` - Pareto Knee (default, recommended)
- `saturation` - Two-stage Saturation
- `constrained` - NE upper bound
- `curvature` - Max curvature
- `knee_log` - Pareto Knee (Log)
- `ideal_point` - Min distance to ideal point
- `weighted_sum` - Min weighted sum
- `max_angle` - Maximum bend angle
- `slope` - Slope target matching (default -1)

## Data

Reads from `TR/datakit/data/ohlcv/` CSV first.
Falls back to yfinance if missing (with warning).
Override path with `DATAKIT_OHLCV_DIR` env var.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
