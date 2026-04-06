from cpm.cpm_core import (
    CriticalPoint,
    extract_local_extrema,
    exceeds_threshold,
    run_cpm,
    compute_normalized_error,
    to_triangle_wave,
)
from cpm.loader import load_prices
from cpm.param_selector import (
    grid_search,
    auto_select,
    format_table,
    print_table,
    METHODS,
)
from cpm.visualize import render_table_image
from cpm.config import GridSearchConfig
