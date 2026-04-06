import numpy as np
import pytest


@pytest.fixture
def sample_prices():
    """Synthetic price series with clear trends and reversals."""
    rng = np.random.RandomState(42)
    t = np.arange(500)
    trend = 100 + 20 * np.sin(2 * np.pi * t / 200) + 0.02 * t
    noise = rng.randn(500) * 0.5
    return trend + noise
