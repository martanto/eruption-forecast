"""Tests for ``eruption_forecast.utils.window``.

Covers the windowing engine shared by RSAM, DSAR, Shannon entropy, and
``LabelBuilder``: daily chunking, per-window metric computation, window
grids, and the entropy kernel.
"""

import numpy as np
import pytest
from scipy.stats import norm

from tests.conftest import build_stream
from eruption_forecast.utils.window import (
    to_safe_array,
    shannon_entropy,
    chunk_daily_data,
    construct_windows,
    get_windows_information,
    calculate_window_metrics,
)


# One sample per minute keeps a full day at 1440 samples and a 10-minute
# window at exactly 10 samples, so every expectation is easy to hand-check.
SAMPLING_RATE = 1 / 60


class TestChunkDailyData:
    def test_full_day_shape(self) -> None:
        data = np.arange(1, 1441, dtype=float)
        chunks = chunk_daily_data(data, sampling_rate=SAMPLING_RATE, window_min=10)
        assert chunks.shape == (144, 10)
        np.testing.assert_array_equal(chunks[0], np.arange(1, 11))

    def test_short_day_is_padded_with_nan(self) -> None:
        data = np.ones(720)
        chunks = chunk_daily_data(data, sampling_rate=SAMPLING_RATE, window_min=10)
        assert chunks.shape == (144, 10)
        assert not np.isnan(chunks[:72]).any()
        assert np.isnan(chunks[72:]).all()

    def test_zero_values_masked(self) -> None:
        data = np.ones(1440)
        data[:5] = 0.0
        chunks = chunk_daily_data(data, sampling_rate=SAMPLING_RATE, window_min=10)
        assert np.isnan(chunks[0, :5]).all()

    def test_zero_values_kept_when_not_masking(self) -> None:
        data = np.ones(1440)
        data[:5] = 0.0
        chunks = chunk_daily_data(
            data, sampling_rate=SAMPLING_RATE, window_min=10, mask_zero_value=False
        )
        assert (chunks[0, :5] == 0.0).all()

    def test_overlap_doubles_window_count(self) -> None:
        data = np.ones(1440)
        chunks = chunk_daily_data(
            data, sampling_rate=SAMPLING_RATE, window_min=10, window_overlap=50
        )
        # Step of 5 samples: (1440 - 10) // 5 + 1 windows.
        assert chunks.shape == (287, 10)

    @pytest.mark.parametrize("overlap", [-1.0, 100.0, 150.0])
    def test_invalid_overlap_raises(self, overlap: float) -> None:
        with pytest.raises(ValueError):
            chunk_daily_data(
                np.ones(1440), sampling_rate=SAMPLING_RATE, window_overlap=overlap
            )


class TestCalculateWindowMetrics:
    def test_mean_over_known_trace(self) -> None:
        stream = build_stream(sampling_rate=SAMPLING_RATE, frequency=0.0, amplitude=0.0)
        trace = stream[0]
        trace.data = np.repeat(np.arange(1, 145, dtype=float), 10)
        series = calculate_window_metrics(trace, window_duration_minutes=10)
        assert len(series) == 144
        np.testing.assert_allclose(series.to_numpy(), np.arange(1, 145))

    def test_absolute_value(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        trace.data = np.full(1440, -3.0)
        series = calculate_window_metrics(trace, absolute_value=True)
        assert (series == 3.0).all()

    def test_insufficient_samples_become_nan(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        data = np.ones(1440)
        data[:8] = np.nan  # Window 0 keeps 2/10 samples, below the 0.3 default.
        trace.data = data
        series = calculate_window_metrics(trace)
        assert np.isnan(series.iloc[0])
        assert series.iloc[1] == 1.0

    def test_value_multiplier(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        trace.data = np.full(1440, 2.0)
        series = calculate_window_metrics(trace, value_multiplier=5.0)
        assert (series == 10.0).all()

    def test_maximum_outlier_removed(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        data = np.tile(np.array([1.0, 2.0] * 10), 72)
        # A huge spike in window 0 would dominate the mean if left in.
        data[0] = 1e6
        trace.data = data
        series = calculate_window_metrics(
            trace, window_duration_minutes=20, remove_outlier_method="maximum"
        )
        assert series.iloc[0] < 2.0

    def test_overlap_timestamps_use_step(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        trace.data = np.ones(1440)
        series = calculate_window_metrics(trace, window_overlap=50)
        step = series.index[1] - series.index[0]
        assert step.total_seconds() == 5 * 60

    def test_index_starts_at_trace_start(self) -> None:
        trace = build_stream(starttime="2025-03-16T00:00:00", sampling_rate=SAMPLING_RATE)[0]
        series = calculate_window_metrics(trace)
        assert str(series.index[0]) == "2025-03-16 00:00:00"


class TestConstructWindows:
    def test_six_hour_grid_over_two_days(self) -> None:
        df = construct_windows("2025-01-01", "2025-01-02", 6, "hours")
        assert len(df) == 8
        assert df.index.name == "datetime"
        assert df.columns.tolist() == ["id", "is_erupted"]
        assert df["id"].tolist() == list(range(8))
        assert (df["is_erupted"] == 0).all()

    def test_minutes_grid_includes_full_last_day(self) -> None:
        df = construct_windows("2025-01-01", "2025-01-02", 10, "minutes")
        assert len(df) == 288
        assert str(df.index[-1]) == "2025-01-02 23:50:00"

    def test_step_larger_than_range_raises(self) -> None:
        with pytest.raises(ValueError):
            construct_windows("2025-01-01", "2025-01-02", 49, "hours")

    def test_invalid_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            construct_windows("2025-01-01", "2025-01-02", 6, "days")  # type: ignore[arg-type]


class TestGetWindowsInformation:
    def test_full_day(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE)[0]
        info = get_windows_information(trace, window_duration_minutes=10)
        assert info["number_of_samples"] == 1440
        assert info["total_windows"] == 144
        assert info["sample_window"] == 144
        assert info["sample_per_window"] == pytest.approx(10.0)

    def test_partial_day(self) -> None:
        trace = build_stream(sampling_rate=SAMPLING_RATE, duration_seconds=43200)[0]
        info = get_windows_information(trace)
        assert info["sample_window"] == 72
        assert info["total_windows"] == 144


class TestShannonEntropyKernel:
    def test_low_energy_returns_nan(self) -> None:
        assert np.isnan(shannon_entropy(np.full(10, 0.01)))

    def test_matches_gaussian_formula(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pdf = norm.pdf(data, loc=data.mean(), scale=data.std())
        expected = -np.sum(pdf * np.log2(pdf))
        assert shannon_entropy(data) == pytest.approx(expected)


class TestToSafeArray:
    def test_masked_values_become_nan(self) -> None:
        masked = np.ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])
        result = to_safe_array(masked)
        assert result.dtype == np.float64
        assert np.isnan(result[1])
        assert result[0] == 1.0

    def test_integer_masked_array(self) -> None:
        masked = np.ma.MaskedArray([1, 2, 3], mask=[False, True, False])
        assert np.isnan(to_safe_array(masked)[1])

    def test_plain_array_cast_to_float(self) -> None:
        result = to_safe_array(np.array([1, 2, 3]))
        assert result.dtype == np.float64
