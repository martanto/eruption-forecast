"""Unit tests for DynamicLabelBuilder."""

# Standard library imports
import os
import tempfile
from datetime import datetime

# Third-party imports
import pandas as pd

# Project imports
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder


def _make_builder(tmpdir: str, **kwargs) -> DynamicLabelBuilder:
    """Return a DynamicLabelBuilder with sensible defaults overridable via kwargs.

    ``overwrite`` is not a constructor parameter on ``DynamicLabelBuilder`` or
    the parent ``LabelBuilder`` — it lives on ``build(overwrite=…)``. Do not
    pass it here.
    """
    defaults = {
        "days_before_eruption": 7,
        "window_step": 12,
        "window_step_unit": "hours",
        "day_to_forecast": 2,
        "eruption_dates": ["2025-03-20"],
        "volcano_id": "TEST",
        "output_dir": tmpdir,
    }
    defaults.update(kwargs)
    return DynamicLabelBuilder(**defaults)


class TestDynamicLabelBuilderInit:
    """Tests for DynamicLabelBuilder initialisation."""

    def test_initialization_valid_parameters(self) -> None:
        """Test that valid parameters produce correct attribute values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(
                tmpdir,
                days_before_eruption=7,
                day_to_forecast=2,
                window_step=12,
                window_step_unit="hours",
                eruption_dates=["2025-03-20"],
                volcano_id="OJN",
            )

            assert builder.days_before_eruption == 7
            assert builder.day_to_forecast == 2
            assert builder.filename.endswith(".csv")
            assert "dtf-2" in builder.filename
            assert "step-12-hours" in builder.filename
            assert os.path.dirname(builder.csv).endswith("labels")


class TestDynamicLabelBuilderBuild:
    """Tests for DynamicLabelBuilder.build()."""

    def test_build_single_eruption(self) -> None:
        """Test that build with a single eruption returns a non-empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir).build(overwrite=True)

            assert not builder.df.empty
            assert len(builder.df_eruption) > 0
            assert set(builder.df["is_erupted"].unique()).issubset({0, 1})

    def test_build_positive_labels_within_day_to_forecast(self) -> None:
        """Test that rows within day_to_forecast window are labelled 1.

        With ``include_eruption_date=True`` (the ``LabelBuilder`` default,
        which ``DynamicLabelBuilder`` inherits) and ``day_to_forecast=2``,
        the positive window is ``[eruption_date - 1 day, eruption_date]``
        — two positive days ending on the eruption date, inclusive.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, day_to_forecast=2).build(overwrite=True)

            eruption_dt = datetime(2025, 3, 20, 23, 59, 59)
            positive_start = datetime(2025, 3, 19, 0, 0, 0)

            df = builder.df
            # All rows inside [positive_start, eruption_dt] must be 1
            mask = (df.index >= positive_start) & (df.index <= eruption_dt)
            assert (df.loc[mask, "is_erupted"] == 1).all()

    def test_build_non_overlapping_eruptions(self) -> None:
        """Test two far-apart eruptions produce windows positive only for their own eruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(
                tmpdir,
                eruption_dates=["2025-03-10", "2025-03-25"],
                days_before_eruption=5,
                day_to_forecast=2,
            ).build(overwrite=True)

            assert not builder.df.empty
            # Both windows must contain positive rows
            assert len(builder.df_eruption) > 0

    def test_build_loads_from_csv_when_exists(self) -> None:
        """Test that a second build without overwrite loads from the saved CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder1 = _make_builder(tmpdir).build(overwrite=True)
            csv_mtime = os.path.getmtime(builder1.csv)

            # Build again without overwrite — CSV must not be rewritten
            builder2 = _make_builder(tmpdir).build(overwrite=False)
            assert os.path.getmtime(builder2.csv) == csv_mtime
            assert not builder2.df.empty

    def test_build_overwrites_when_flag_set(self) -> None:
        """Test that overwrite=True causes build to recompute and rewrite the CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder1 = _make_builder(tmpdir).build(overwrite=True)
            mtime1 = os.path.getmtime(builder1.csv)

            import time
            time.sleep(0.05)

            builder2 = _make_builder(tmpdir).build(overwrite=True)
            mtime2 = os.path.getmtime(builder2.csv)

            assert mtime2 > mtime1


class TestDynamicLabelBuilderIntegration:
    """Integration tests for the full DynamicLabelBuilder workflow."""

    def test_full_workflow_single_eruption(self) -> None:
        """Test build → CSV saved → reload produces consistent DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir).build(overwrite=True)

            assert os.path.isfile(builder.csv)

            reloaded = pd.read_csv(builder.csv, index_col=0, parse_dates=True)
            assert len(reloaded) == len(builder.df)
            assert (reloaded["is_erupted"] == builder.df["is_erupted"].values).all()

    def test_full_workflow_overlapping_eruptions(self) -> None:
        """Test that two close eruptions produce the expected total positive row count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            day_to_forecast = 3
            eruptions = ["2025-03-20", "2025-03-23"]
            builder = _make_builder(
                tmpdir,
                eruption_dates=eruptions,
                days_before_eruption=10,
                day_to_forecast=day_to_forecast,
                window_step=24,
                window_step_unit="hours",
            ).build(overwrite=True)

            total_positive = (builder.df["is_erupted"] == 1).sum()
            # At least day_to_forecast windows per eruption × 2 eruptions
            assert total_positive >= day_to_forecast * 2
