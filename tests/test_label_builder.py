"""Unit tests for the label building module.

Tests for LabelBuilder and LabelData classes.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.label.label_data import LabelData


def _make_builder(tmpdir, **kwargs) -> LabelBuilder:
    """Return a LabelBuilder with sensible defaults, overridable via kwargs."""
    defaults = dict(
        start_date="2020-01-01",
        end_date="2020-01-15",
        window_step=12,
        window_step_unit="hours",
        day_to_forecast=2,
        eruption_dates=["2020-01-10"],
        volcano_id="TEST_001",
        output_dir=tmpdir,
    )
    defaults.update(kwargs)
    return LabelBuilder(**defaults)


class TestLabelBuilder:
    def test_initialization_valid_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            assert builder.start_date == datetime(2020, 1, 1, 0, 0, 0)
            assert builder.window_step == 12
            assert builder.window_step_unit == "hours"
            assert builder.day_to_forecast == 2
            assert builder.volcano_id == "TEST_001"

    def test_validation_start_date_after_end_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                _make_builder(
                    tmpdir,
                    start_date="2020-01-15",
                    end_date="2020-01-01",
                )

    def test_validation_invalid_window_step_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((ValueError, TypeError)):
                _make_builder(tmpdir, window_step_unit="days")

    def test_build_creates_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            builder.build()

            assert not builder.df.empty
            assert "id" in builder.df.columns
            assert "is_erupted" in builder.df.columns
            assert isinstance(builder.df.index, pd.DatetimeIndex)
            assert (builder.df["is_erupted"] == 1).any()
            assert (builder.df["is_erupted"] == 0).any()

    def test_build_with_no_eruptions_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                _make_builder(
                    tmpdir,
                    eruption_dates=["2019-12-15"],
                ).build()

    def test_build_labels_correctly_with_day_to_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, window_step=24, day_to_forecast=2)
            builder.build()
            df = builder.df_eruption
            assert len(df) > 0

    def test_save_creates_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            builder.build().save()

            assert os.path.exists(builder.csv)
            assert "label_2020-01-01_2020-01-15" in builder.csv
            assert "step-12-hours" in builder.csv
            assert "dtf-2" in builder.csv

    def test_df_property_raises_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir)
            with pytest.raises(ValueError):
                _ = builder.df


class TestLabelData:
    def _create_label_csv(self, tmpdir) -> str:
        builder = _make_builder(tmpdir)
        builder.build().save()
        return builder.csv

    def test_initialization_with_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = self._create_label_csv(tmpdir)
            ld = LabelData(csv)
            assert ld.window_step == 12
            assert ld.window_unit == "hours"
            assert ld.day_to_forecast == 2
            assert ld.start_date_str == "2020-01-01"
            assert ld.end_date_str == "2020-01-15"

    def test_validation_file_not_found(self) -> None:
        with pytest.raises(ValueError):
            LabelData(
                "/nonexistent/path/label_2020-01-01_2020-01-15_step-12-hours_dtf-2.csv"
            )

    def test_validation_invalid_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = os.path.join(
                tmpdir, "invalid_2020-01-01_2020-01-15_step-12-hours_dtf-2.csv"
            )
            Path(invalid_file).touch()
            with pytest.raises(ValueError):
                LabelData(invalid_file)

    def test_parameters_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = self._create_label_csv(tmpdir)
            ld = LabelData(csv)
            params = ld.parameters
            assert params["window_step"] == 12
            assert params["window_unit"] == "hours"
            assert params["day_to_forecast"] == 2
            assert params["start_date_str"] == "2020-01-01"
            assert params["end_date_str"] == "2020-01-15"


class TestLabelIntegration:
    def test_full_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-31",
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10", "2020-01-20"],
                volcano_id="TEST_VOLCANO",
                output_dir=tmpdir,
            )
            builder.build().save()

            label_data = LabelData(builder.csv)

            assert label_data.window_step == builder.window_step
            assert label_data.day_to_forecast == builder.day_to_forecast

            df = label_data.df
            assert not df.empty
            assert "id" in df.columns
            assert "is_erupted" in df.columns
            assert (df["is_erupted"] == 1).sum() > 0
