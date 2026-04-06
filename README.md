# cpm-param-selector

Automatic parameter selection for the **Critical Point Model (CPM)** in financial time series.

CPM identifies turning points (peaks and troughs) in price data by filtering noise through two parameters:
- **P**: price oscillation threshold (e.g., 0.06 = 6%)
- **T**: minimum time duration between turning points (in days)

Choosing the right (P, T) is critical — too aggressive filters real signals, too conservative keeps noise. This tool automates that selection.

## Background

Based on:
- Bao (2008), *"A generalized model for financial time series representation and prediction"*
- Bao & Yang (2008), *"Intelligent stock trading system by turning point confirming and probabilistic reasoning"*
- Lin et al. (2011), *"Intelligent stock trading system based on improved technical analysis and Echo State Network"*

Bao (2008) proposed a grid search over (P, T) for IBM (Table 1) but did not provide a selection criterion. This tool extends the grid search to multi-asset ETFs and implements five automatic selection methods.

## Project Structure

```
cpm-param-selector/
├── cpm/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Grid search range configuration
│   ├── cpm_core.py          # CPM algorithm (extrema, 3-point unit, error, triangle wave)
│   ├── loader.py            # Price data loading (yfinance)
│   ├── param_selector.py    # Grid search + 5 auto-selection methods
│   └── visualize.py         # Table image rendering
├── scripts/
│   └── run_param_search.py  # CLI script for running grid search
└── tests/
    └── test_cpm.py          # Unit tests
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cpm import load_prices, auto_select, print_table

# Load price data
prices = load_prices("SPY")

# Grid search table (Bao 2008 Table 1 style)
print_table(prices, ticker="SPY")

# Auto-select (P, T)
P, T = auto_select(prices, method="knee")        # Pareto Knee (default)
P, T = auto_select(prices, method="saturation")  # Two-stage Saturation
```

### All methods

```python
from cpm import auto_select

# Primary methods
P, T = auto_select(prices, method="knee")          # 4. Pareto Knee (Linear)
P, T = auto_select(prices, method="saturation")    # 1. Two-stage Saturation

# Additional methods (for comparison)
P, T = auto_select(prices, method="constrained")   # 2. NE upper bound
P, T = auto_select(prices, method="curvature")     # 3. Max curvature
P, T = auto_select(prices, method="knee_log")      # 5. Pareto Knee (Log)
```

### Grid search table output

```
         0.02      0.04      0.06      0.08      0.10
T
4   0.7%/1101  0.8%/905  0.8%/869  0.8%/865  0.8%/859
6    0.8%/927  1.0%/635  1.0%/583  1.0%/571  1.0%/561
8    0.9%/845  1.2%/493  1.2%/431  1.2%/409  1.2%/397
```

### Render table as image

```python
from cpm import grid_search, render_table_image

df = grid_search(prices)
render_table_image(df, ticker="SPY", save_path="SPY_table.png")
```

## How It Works

### CPM (Critical Point Model)

1. Extract all local maxima/minima from price series
2. Process 3-point units iteratively, classifying into 4 cases:
   - **Case 1**: Both sides exceed threshold -> preserve turning point
   - **Case 2**: Left exceeds, right doesn't -> look ahead
   - **Case 3**: Right exceeds, left doesn't -> skip
   - **Case 4**: Neither exceeds -> remove middle point
3. Output: filtered list of significant turning points

### Parameter Selection

The trade-off: fewer points = more noise removed, but higher approximation error.

- **Pareto Knee** finds where reducing points further causes error to spike — the point of maximum perpendicular distance from the diagonal on the normalized Pareto front.
- **Saturation** finds where increasing P (or T) stops reducing the number of points — indicating all noise has been filtered.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
