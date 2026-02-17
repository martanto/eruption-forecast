"""Unit tests for the label building module.

Tests for LabelBuilder and LabelData classes to ensure proper functionality
after Phase 2 refactoring.
"""

# Standard library imports
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Third party imports
import pandas as pd
import pytest

from eruption_forecast.label.label_data import LabelData

# Project imports
from eruption_forecast.label.label_builder import LabelBuilder


class TestLabelBuilder:
    """Test cases for LabelBuilder class."""

    def test_initialization_valid_parameters(self) -> None:
        """Test LabelBuilder initialization with valid parameters."""
        builder = LabelBuilder(
            start_date="2020-01-01",
            end_date="2020-01-15",
            window_size=1,
            window_step=12,
            window_step_unit="hours",
            day_to_forecast=2,
            eruption_dates=["2020-01-10"],
            volcano_id="TEST_001",
        )

        assert builder.start_date == datetime(2020, 1, 1, 0, 0, 0)
        assert builder.end_date == datetime(2020, 1, 15, 23, 59, 59)
        assert builder.window_size == 1
        assert builder.window_step == 12
        assert builder.window_step_unit == "hours"
        assert builder.day_to_forecast == 2
        assert builder.eruption_dates == ["2020-01-10"]
        assert builder.volcano_id == "TEST_001"

    def test_validation_start_date_after_end_date(self) -> None:
        """Test that ValueError is raised when start_date >= end_date."""
        with pytest.raises(ValueError, match="start_date must be less than end_date"):
            LabelBuilder(
                start_date="2020-01-15",
                end_date="2020-01-01",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
            )

    def test_validation_insufficient_date_range(self) -> None:
        """Test that ValueError is raised when date range < 7 days."""
        with pytest.raises(ValueError, match="must be >= 7 days"):
            LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-05",  # Only 4 days
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-03"],
                volcano_id="TEST_001",
            )

    def test_validation_invalid_window_step_unit(self) -> None:
        """Test that ValueError is raised for invalid window_step_unit."""
        with pytest.raises(ValueError, match="window_step_unit must be one of"):
            LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="days",  # Invalid unit
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
            )

    def test_validation_zero_window_size(self) -> None:
        """Test that ValueError is raised for window_size <= 0."""
        with pytest.raises(ValueError, match="window_size must be > 0"):
            LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=0,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
            )

    def test_build_creates_labels(self) -> None:
        """Test that build() creates label DataFrame with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )

            builder.build()

            # Check DataFrame structure
            assert not builder.df.empty
            assert "id" in builder.df.columns
            assert "is_erupted" in builder.df.columns
            assert isinstance(builder.df.index, pd.DatetimeIndex)

            # Check that some labels are positive
            assert (builder.df["is_erupted"] == 1).any()
            assert (builder.df["is_erupted"] == 0).any()

    def test_build_with_no_eruptions_in_range(self) -> None:
        """Test that ValueError is raised when no eruptions in date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No eruption between start date"):
                builder = LabelBuilder(
                    start_date="2020-01-01",
                    end_date="2020-01-15",
                    window_size=1,
                    window_step=12,
                    window_step_unit="hours",
                    day_to_forecast=2,
                    eruption_dates=["2019-12-15"],  # Before range
                    volcano_id="TEST_001",
                    output_dir=tmpdir,
                )
                builder.build()

    def test_build_labels_correctly_with_day_to_forecast(self) -> None:
        """Test that labels are set correctly based on day_to_forecast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=24,  # Daily windows
                window_step_unit="hours",
                day_to_forecast=2,  # Label 2 days before eruption
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )

            builder.build()

            # Check that windows from 2020-01-08 onwards are labeled
            # (day_to_forecast=2 means start labeling 2 days before eruption)
            df = builder.df_eruption
            assert len(df) > 0

    def test_save_creates_csv_file(self) -> None:
        """Test that save() creates CSV file with correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )

            builder.build().save()

            # Check that CSV file exists
            assert os.path.exists(builder.csv)

            # Check that eruption_dates.csv exists
            eruption_dates_file = os.path.join(builder.label_dir, "eruption_dates.csv")
            assert os.path.exists(eruption_dates_file)

            # Check filename format
            assert (
                "label_2020-01-01_2020-01-15_ws-1_step-12-hours_dtf-2.csv"
                in builder.csv
            )

    def test_df_property_raises_before_build(self) -> None:
        """Test that accessing df property before build() raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )

            with pytest.raises(ValueError, match="Please call 'build' method first"):
                _ = builder.df


class TestLabelData:
    """Test cases for LabelData class."""

    def test_initialization_with_valid_file(self) -> None:
        """Test LabelData initialization with valid filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a label file first
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )
            builder.build().save()

            # Load with LabelData
            label_data = LabelData(builder.csv)

            assert label_data.window_size == 1
            assert label_data.window_step == 12
            assert label_data.window_unit == "hours"
            assert label_data.day_to_forecast == 2
            assert label_data.start_date_str == "2020-01-01"
            assert label_data.end_date_str == "2020-01-15"

    def test_validation_file_not_found(self) -> None:
        """Test that ValueError is raised when file doesn't exist."""
        with pytest.raises(ValueError, match="Label file not found"):
            LabelData(
                "/nonexistent/path/label_2020-01-01_2020-01-15_ws-1_step-12-hours_dtf-2.csv"
            )

    def test_validation_invalid_prefix(self) -> None:
        """Test that ValueError is raised for invalid filename prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with invalid name
            invalid_file = os.path.join(
                tmpdir, "invalid_2020-01-01_2020-01-15_ws-1_step-12-hours_dtf-2.csv"
            )
            Path(invalid_file).touch()

            with pytest.raises(ValueError, match="Filename should start with"):
                LabelData(invalid_file)

    def test_validation_invalid_extension(self) -> None:
        """Test that ValueError is raised for invalid file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with invalid extension
            invalid_file = os.path.join(
                tmpdir, "label_2020-01-01_2020-01-15_ws-1_step-12-hours_dtf-2.txt"
            )
            Path(invalid_file).touch()

            with pytest.raises(ValueError, match="Label file extension is invalid"):
                LabelData(invalid_file)

    def test_validation_invalid_part_count(self) -> None:
        """Test that ValueError is raised for incorrect number of filename parts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with too few parts
            invalid_file = os.path.join(tmpdir, "label_2020-01-01_ws-1.csv")
            Path(invalid_file).touch()

            with pytest.raises(ValueError, match="Label filename is invalid"):
                LabelData(invalid_file)

    def test_parameters_property(self) -> None:
        """Test that parameters property returns all extracted values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a label file first
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-15",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10"],
                volcano_id="TEST_001",
                output_dir=tmpdir,
            )
            builder.build().save()

            # Load with LabelData
            label_data = LabelData(builder.csv)
            params = label_data.parameters

            assert params["window_size"] == 1
            assert params["window_step"] == 12
            assert params["window_unit"] == "hours"
            assert params["day_to_forecast"] == 2
            assert params["start_date_str"] == "2020-01-01"
            assert params["end_date_str"] == "2020-01-15"


class TestLabelIntegration:
    """Integration tests for label building workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: build, save, load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Build labels
            builder = LabelBuilder(
                start_date="2020-01-01",
                end_date="2020-01-31",
                window_size=1,
                window_step=12,
                window_step_unit="hours",
                day_to_forecast=2,
                eruption_dates=["2020-01-10", "2020-01-20"],
                volcano_id="TEST_VOLCANO",
                output_dir=tmpdir,
            )
            builder.build().save()

            # Step 2: Load with LabelData
            label_data = LabelData(builder.csv)

            # Step 3: Verify consistency
            assert label_data.window_size == builder.window_size
            assert label_data.window_step == builder.window_step
            assert label_data.day_to_forecast == builder.day_to_forecast

            # Step 4: Check DataFrame
            df = label_data.df
            assert not df.empty
            assert "id" in df.columns
            assert "is_erupted" in df.columns

            # Step 5: Verify some windows are labeled
            erupted_count = (df["is_erupted"] == 1).sum()
            assert erupted_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
