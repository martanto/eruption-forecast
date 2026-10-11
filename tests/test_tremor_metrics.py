"""Tests for the RSAM, DSAR, and Shannon-entropy tremor calculators.

Every test runs on a synthetic one-day 20 Hz ``Stream`` built by
``conftest.build_stream`` so the expected values can be derived analytically:
the mean absolute value of ``A * sin(...)`` is ``2A / pi``.
"""

import numpy as np
import pandas as pd
import pytest
from obspy import Stream

from tests.conftest import STREAM_AMPLITUDE, build_stream
from eruption_forecast.tremor.dsar import DSAR
from eruption_forecast.tremor.rsam import RSAM
from eruption_forecast.tremor.shannon_entropy import ShannonEntropy


WINDOWS_PER_DAY = 144


class TestRSAM:
    def test_unfiltered_mean_matches_analytic_value(self, synthetic_stream) -> None:
        series = RSAM(synthetic_stream).calculate()
        expected = 2 * STREAM_AMPLITUDE / np.pi
        assert len(series) == WINDOWS_PER_DAY
        assert series.median() == pytest.approx(expected, rel=0.01)

    def test_in_band_filter_preserves_amplitude(self, synthetic_stream) -> None:
        rsam = RSAM(synthetic_stream).apply_filter(2.0, 5.0)
        series = rsam.calculate()
        assert rsam.is_filtered is True
        assert series.median() == pytest.approx(2 * STREAM_AMPLITUDE / np.pi, rel=0.02)

    def test_out_of_band_filter_attenuates(self, synthetic_stream) -> None:
        # The 3 Hz signal sits well outside 8-9.5 Hz, so almost nothing passes.
        series = RSAM(synthetic_stream).apply_filter(8.0, 9.5).calculate()
        assert series.median() < 0.01 * STREAM_AMPLITUDE

    def test_index_is_ten_minute_grid(self, synthetic_stream) -> None:
        series = RSAM(synthetic_stream).calculate()
        assert series.index[0] == pd.Timestamp("2025-01-01 00:00")
        assert (pd.Series(series.index).diff().dropna() == pd.Timedelta("10min")).all()

    def test_value_multiplier_scales_output(self, synthetic_stream) -> None:
        base = RSAM(synthetic_stream.copy()).calculate()
        scaled = RSAM(synthetic_stream.copy()).calculate(value_multiplier=10.0)
        np.testing.assert_allclose(scaled.to_numpy(), base.to_numpy() * 10.0)

    def test_half_day_trace_pads_with_nan_without_interpolation(self) -> None:
        stream = build_stream(duration_seconds=43200)
        series = RSAM(stream).calculate(interpolate=False)
        assert len(series) == WINDOWS_PER_DAY
        assert series.iloc[:72].notna().all()
        assert series.iloc[72:].isna().all()

    def test_series_is_stored_on_instance(self, synthetic_stream) -> None:
        rsam = RSAM(synthetic_stream)
        series = rsam.calculate()
        assert rsam.series is series
        assert rsam.start_datetime_str == "2025-01-01"


class TestDSAR:
    def test_ratio_of_two_streams(self) -> None:
        ratio = DSAR().calculate(
            build_stream(amplitude=1000.0), build_stream(amplitude=500.0)
        )
        assert len(ratio) == WINDOWS_PER_DAY
        np.testing.assert_allclose(ratio.to_numpy(), 2.0, rtol=1e-6)

    def test_accepts_precomputed_series(self) -> None:
        index = pd.date_range("2025-01-01", periods=4, freq="10min")
        first = pd.Series([2.0, 4.0, 6.0, 8.0], index=index)
        second = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        dsar = DSAR()
        ratio = dsar.calculate(first, second)
        assert ratio.tolist() == [2.0, 2.0, 2.0, 2.0]
        assert dsar.first_dsar is first
        assert dsar.second_dsar is second
        assert dsar.series is ratio

    def test_value_multiplier(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="10min")
        ratio = DSAR().calculate(
            pd.Series([4.0, 6.0], index=index),
            pd.Series([2.0, 2.0], index=index),
            value_multiplier=0.5,
        )
        assert ratio.tolist() == [1.0, 1.5]

    def test_interpolates_gaps(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="10min")
        ratio = DSAR().calculate(
            pd.Series([2.0, np.nan, 6.0], index=index),
            pd.Series([1.0, 1.0, 1.0], index=index),
        )
        assert ratio.tolist() == [2.0, 4.0, 6.0]

    def test_no_interpolation_keeps_nan(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="10min")
        ratio = DSAR().calculate(
            pd.Series([2.0, np.nan, 6.0], index=index),
            pd.Series([1.0, 1.0, 1.0], index=index),
            interpolate=False,
        )
        assert np.isnan(ratio.iloc[1])


class TestShannonEntropy:
    def test_default_band(self, synthetic_stream) -> None:
        entropy = ShannonEntropy(synthetic_stream)
        assert (entropy.freqmin, entropy.freqmax) == (1.0, 16.0)

    def test_filter_sets_band(self, synthetic_stream) -> None:
        entropy = ShannonEntropy(synthetic_stream).filter(1, 8)
        assert (entropy.freqmin, entropy.freqmax) == (1.0, 8.0)

    def test_filter_min_above_max_raises(self, synthetic_stream) -> None:
        with pytest.raises(ValueError):
            ShannonEntropy(synthetic_stream).filter(8.0, 1.0)

    def test_calculate_returns_finite_daily_grid(self, synthetic_stream) -> None:
        series = ShannonEntropy(synthetic_stream).filter(1.0, 8.0).calculate()
        assert len(series) == WINDOWS_PER_DAY
        assert np.isfinite(series.dropna()).all()
        assert series.notna().sum() >= WINDOWS_PER_DAY - 1

    def test_calculate_does_not_mutate_trace(self, synthetic_stream: Stream) -> None:
        original = synthetic_stream[0].data.copy()
        ShannonEntropy(synthetic_stream).filter(1.0, 8.0).calculate()
        np.testing.assert_array_equal(synthetic_stream[0].data, original)
