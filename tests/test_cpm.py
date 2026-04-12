"""Tests for CPM and parameter selection."""

import numpy as np
import pytest

from cpm.cpm_core import (
    CriticalPoint,
    compute_normalized_error,
    exceeds_threshold,
    extract_local_extrema,
    run_cpm,
    to_triangle_wave,
)
from cpm.param_selector import auto_select, grid_search, METHODS


class TestCPM:
    def test_extract_alternating(self):
        rng = np.random.RandomState(123)
        prices = 50 + np.cumsum(rng.randn(200))
        extrema = extract_local_extrema(prices)
        for i in range(1, len(extrema)):
            assert extrema[i].is_max != extrema[i - 1].is_max

    def test_cpm_reduces_points(self, sample_prices):
        extrema = extract_local_extrema(sample_prices)
        tp = run_cpm(sample_prices, P=0.06, T=8)
        assert len(tp) < len(extrema)
        assert len(tp) >= 2

    def test_higher_T_fewer_points(self, sample_prices):
        tp_low = run_cpm(sample_prices, P=0.06, T=4)
        tp_high = run_cpm(sample_prices, P=0.06, T=16)
        assert len(tp_high) <= len(tp_low)

    def test_normalized_error_perfect(self):
        prices = np.array([10.0, 15.0, 8.0, 20.0, 5.0])
        tps = [
            CriticalPoint(0, 10.0, False),
            CriticalPoint(1, 15.0, True),
            CriticalPoint(2, 8.0, False),
            CriticalPoint(3, 20.0, True),
            CriticalPoint(4, 5.0, False),
        ]
        assert compute_normalized_error(prices, tps) < 0.1

    def test_triangle_wave_range(self):
        tps = [
            CriticalPoint(0, 100.0, False),
            CriticalPoint(50, 150.0, True),
            CriticalPoint(99, 80.0, False),
        ]
        wave = to_triangle_wave(tps, 100)
        assert np.all(wave >= -1.0)
        assert np.all(wave <= 1.0)

    def test_triangle_wave_endpoints(self):
        tps = [
            CriticalPoint(0, 100.0, False),
            CriticalPoint(10, 120.0, True),
            CriticalPoint(20, 90.0, False),
        ]
        wave = to_triangle_wave(tps, 21)
        assert wave[0] == pytest.approx(-1.0)
        assert wave[10] == pytest.approx(1.0)
        assert wave[20] == pytest.approx(-1.0)


class TestParamSelector:
    def test_grid_search_shape(self, sample_prices):
        df = grid_search(sample_prices)
        assert len(df) == 9 * 7  # 9 P values x 7 T values
        assert "error_raw" in df.columns
        assert "error_pct" in df.columns

    def test_all_methods_run(self, sample_prices):
        for method in METHODS:
            P, T = auto_select(sample_prices, method=method)
            assert 0.02 <= P <= 0.18
            assert 4 <= T <= 16

    def test_invalid_method(self, sample_prices):
        with pytest.raises(ValueError):
            auto_select(sample_prices, method="invalid")


class TestLoader:
    def test_reads_csv_when_exists(self, tmp_path, monkeypatch):
        """reads local CSV when it exists"""
        import cpm.loader as loader

        csv = tmp_path / "TEST.csv"
        csv.write_text("Date,Open,High,Low,Close,Volume,Symbol\n2024-01-01,100,110,90,105,1000,TEST\n2024-01-02,105,115,95,110,1200,TEST\n")

        monkeypatch.setattr(loader, "_get_ohlcv_dir", lambda: tmp_path)
        prices = loader.load_prices("TEST")
        assert len(prices) == 2
        assert prices[0] == pytest.approx(105.0)
        assert prices[1] == pytest.approx(110.0)

    def test_fallback_when_no_csv(self, tmp_path, monkeypatch, mocker):
        """falls back to yfinance with warning when no CSV"""
        import cpm.loader as loader

        monkeypatch.setattr(loader, "_get_ohlcv_dir", lambda: tmp_path)

        mock_df = mocker.MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = lambda self, key: mocker.MagicMock(
            values=np.array([[100.0, 200.0]])
        )
        mocker.patch("yfinance.download", return_value=mock_df)

        with pytest.warns(UserWarning, match="CSV not found"):
            prices = loader.load_prices("FAKE_TICKER")
        assert len(prices) > 0

    def test_skips_broken_csv(self, tmp_path, monkeypatch, mocker):
        """falls back with warning when CSV has no Close column"""
        import cpm.loader as loader

        csv = tmp_path / "BROKEN.csv"
        csv.write_text("Date,Open,High,Low\n2024-01-01,100,110,90\n")

        monkeypatch.setattr(loader, "_get_ohlcv_dir", lambda: tmp_path)

        mock_df = mocker.MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = lambda self, key: mocker.MagicMock(
            values=np.array([[100.0]])
        )
        mocker.patch("yfinance.download", return_value=mock_df)

        with pytest.warns(UserWarning, match="missing 'Close' column"):
            prices = loader.load_prices("BROKEN")
        assert len(prices) > 0


class TestPublicAPI:
    def test_import_all(self):
        from cpm import (
            CriticalPoint, extract_local_extrema, run_cpm,
            compute_normalized_error, to_triangle_wave,
            load_prices, grid_search, auto_select, format_table,
            print_table, render_table_image, METHODS, GridSearchConfig,
        )
