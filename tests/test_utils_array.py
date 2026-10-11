"""Tests for ``eruption_forecast.utils.array``.

Covers outlier removal, completeness counting, the seed-probability
aggregation kernel, and per-seed CSV persistence.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import build_stream
from eruption_forecast.utils.array import (
    filter_nans,
    remove_outliers,
    get_completeness,
    mask_zero_values,
    count_valid_values,
    save_forecast_seed,
    confidence_interval,
    remove_maximum_outlier,
    compute_model_probabilities,
)


class TestMaskZeroValues:
    def test_removes_zeros(self) -> None:
        np.testing.assert_array_equal(
            mask_zero_values(np.array([0.0, 1.0, 0.0, 2.0])), [1.0, 2.0]
        )

    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError):
            mask_zero_values([0, 1])  # type: ignore[arg-type]


class TestCountValidValues:
    def test_excludes_zero_and_nan(self) -> None:
        assert count_valid_values(np.array([0.0, np.nan, 1.0, 2.0])) == 2

    def test_rejects_non_array(self) -> None:
        with pytest.raises(TypeError):
            count_valid_values([1, 2])  # type: ignore[arg-type]


def test_filter_nans() -> None:
    np.testing.assert_array_equal(
        filter_nans(np.array([np.nan, 1.0, np.nan, 3.0])), [1.0, 3.0]
    )


class TestRemoveMaximumOutlier:
    def test_removes_single_spike(self) -> None:
        # Non-zero spread so the z-score is defined, then one large spike.
        data = np.concatenate([np.tile([1.0, 2.0], 10), [100.0]])
        result = remove_maximum_outlier(data)
        assert 100.0 not in result
        assert len(result) == len(data) - 1

    def test_no_outlier_returns_unchanged(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(remove_maximum_outlier(data), data)

    def test_masks_zero_by_default(self) -> None:
        result = remove_maximum_outlier(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_all_zero_returns_empty(self) -> None:
        assert len(remove_maximum_outlier(np.zeros(5))) == 0

    def test_does_not_mutate_input(self) -> None:
        data = np.array([0.0, 1.0, 2.0])
        remove_maximum_outlier(data)
        np.testing.assert_array_equal(data, [0.0, 1.0, 2.0])


class TestRemoveOutliers:
    @pytest.fixture()
    def spiky(self) -> np.ndarray:
        data = np.tile([1.0, 2.0], 20)
        return np.concatenate([data, [500.0]])

    def test_removes_outliers(self, spiky: np.ndarray) -> None:
        result = remove_outliers(spiky)
        assert 500.0 not in result
        assert len(result) == len(spiky) - 1

    def test_return_outliers(self, spiky: np.ndarray) -> None:
        np.testing.assert_array_equal(
            remove_outliers(spiky, return_outliers=True), [500.0]
        )

    def test_constant_data_returns_data(self) -> None:
        # The docstring says an empty array; the code returns the input.
        data = np.full(5, 3.0)
        np.testing.assert_array_equal(remove_outliers(data), data)

    def test_constant_data_has_no_outliers(self) -> None:
        assert len(remove_outliers(np.full(5, 3.0), return_outliers=True)) == 0

    def test_nan_filtered(self) -> None:
        result = remove_outliers(np.array([1.0, np.nan, 2.0]))
        assert not np.isnan(result).any()

    def test_non_positive_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            remove_outliers(np.array([1.0, 2.0]), outlier_threshold=0)


class TestGetCompleteness:
    def test_full_day(self) -> None:
        stream = build_stream(sampling_rate=1 / 60, amplitude=1.0, frequency=0.001)
        stream[0].data = np.ones(1440)
        assert get_completeness(stream) == pytest.approx(1.0)

    def test_half_day_trace(self) -> None:
        trace = build_stream(sampling_rate=1 / 60, duration_seconds=43200)[0]
        trace.data = np.ones(720)
        assert get_completeness(trace) == pytest.approx(0.5)

    def test_rejects_other_types(self) -> None:
        with pytest.raises(TypeError):
            get_completeness(np.ones(10))  # type: ignore[arg-type]


class TestComputeModelProbabilities:
    def test_aggregates(self) -> None:
        probabilities = np.array([[0.2, 0.4], [0.9, 0.7]])
        predictions = np.array([[0, 0], [1, 0]])

        mean_p, std_p, mean_pred, confidence = compute_model_probabilities(
            probabilities, predictions
        )

        np.testing.assert_allclose(mean_p, [0.3, 0.8])
        np.testing.assert_allclose(std_p, [0.1, 0.1])
        np.testing.assert_allclose(mean_pred, [0.0, 0.5])
        np.testing.assert_allclose(confidence, [0.0, 1.96 * np.sqrt(0.25 / 2)])

    def test_confidence_interval_bounds(self) -> None:
        # Unanimous votes carry no uncertainty; a 50/50 split carries the most.
        ci = confidence_interval(np.array([0.0, 0.5, 1.0]), total_predictions=100)
        assert ci[0] == 0.0
        assert ci[2] == 0.0
        assert ci[1] == pytest.approx(1.96 * 0.05)


class TestSaveForecastSeed:
    def test_writes_csv(self, tmp_path) -> None:
        save_forecast_seed(
            output_dir=str(tmp_path),
            random_state=7,
            probabilities=np.array([0.1, 0.8]),
            predictions=np.array([0, 1]),
        )
        df = pd.read_csv(tmp_path / "p_00007.csv", index_col="label_id")
        assert df.columns.tolist() == ["p_no_eruption", "p_eruption", "prediction"]
        np.testing.assert_allclose(df["p_no_eruption"], [0.9, 0.2])
        assert df["prediction"].tolist() == [0, 1]

    def test_skips_existing_without_overwrite(self, tmp_path) -> None:
        path = tmp_path / "p_00001.csv"
        path.write_text("sentinel", encoding="utf-8")
        save_forecast_seed(
            str(tmp_path), 1, np.array([0.5]), np.array([1]), overwrite=False
        )
        assert path.read_text(encoding="utf-8") == "sentinel"

    def test_overwrite_replaces_existing(self, tmp_path) -> None:
        path = tmp_path / "p_00001.csv"
        path.write_text("sentinel", encoding="utf-8")
        save_forecast_seed(
            str(tmp_path), 1, np.array([0.5]), np.array([1]), overwrite=True
        )
        assert os.path.getsize(path) > len("sentinel")
