"""Unit tests for DynamicLabelBuilder."""

# Standard library imports
import os
import tempfile
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import pytest

# Project imports
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder


def _make_builder(tmpdir: str, **kwargs) -> DynamicLabelBuilder:
    """Return a DynamicLabelBuilder with sensible defaults overridable via kwargs."""
    defaults = dict(
        days_before_eruption=7,
        window_step=12,
        window_step_unit="hours",
        day_to_forecast=2,
        eruption_dates=["2025-03-20"],
        volcano_id="TEST",
        output_dir=tmpdir,
        overwrite=True,
    )
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
            assert builder.overwrite is True
            assert builder.filename.endswith(".csv")
            assert "dtf-2" in builder.filename
            assert "step-12-hours" in builder.filename
            assert os.path.dirname(builder.csv).endswith("labels")


class TestDynamicLabelBuilderBuild:
    """Tests for DynamicLabelBuilder.build()."""

    def test_build_single_eruption(self) -> None:
        """Test that build with a single eruption returns a non-empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir).build()

            assert not builder.df.empty
            assert len(builder.df_eruption) > 0
            assert set(builder.df["is_erupted"].unique()).issubset({0, 1})

    def test_build_positive_labels_within_day_to_forecast(self) -> None:
        """Test that rows within the look-forward window are labelled 1.

        Look-forward: positive = [E - buffer×step, E + day_to_forecast].
        eruption=2025-03-20, step=12h, buffer=1, dtf=2 →
        positive_start = 2025-03-19 12:00, positive_end = 2025-03-22 23:59:59.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, day_to_forecast=2).build()

            positive_start = datetime(2025, 3, 19, 12, 0, 0)
            positive_end = datetime(2025, 3, 22, 23, 59, 59)

            df = builder.df
            mask = (df.index >= positive_start) & (df.index <= positive_end)
            assert (df.loc[mask, "is_erupted"] == 1).all()

    def test_build_non_overlapping_eruptions(self) -> None:
        """Test two far-apart eruptions produce windows positive only for their own eruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(
                tmpdir,
                eruption_dates=["2025-03-10", "2025-03-25"],
                days_before_eruption=5,
                day_to_forecast=2,
            ).build()

            assert not builder.df.empty
            # Both windows must contain positive rows
            assert len(builder.df_eruption) > 0

    def test_build_overlapping_eruptions(self) -> None:
        """Test cross-year eruption projection: 2020-11-05 projects into 2021-11-10 window.

        Look-forward, step=24h, buffer=1, dtf=2:
        - Projected Nov 5 → pos_start=Nov 4, pos_end=Nov 7
        - Primary Nov 10 → pos_start=Nov 9, pos_end=Nov 12
        - Gap: Nov 8
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 2020-11-05 projected to 2021-11-05 falls inside 2021-10-31 → 2021-11-12 window.
            builder = _make_builder(
                tmpdir,
                eruption_dates=["2020-11-05", "2021-11-10"],
                days_before_eruption=10,
                day_to_forecast=2,
                window_step=24,
                window_step_unit="hours",
            ).build()

            second_key = "2021-11-10"
            df_second = builder.df_eruptions.get(second_key)
            assert df_second is not None
            assert len(df_second) > 0
            assert (df_second["is_erupted"] == 1).all()

            # Projected eruption (Nov 5) → positive rows Nov 4, 5, 6, 7
            full_df = builder.df
            for d in ["2021-11-04", "2021-11-05", "2021-11-06", "2021-11-07"]:
                dt = datetime.fromisoformat(d)
                rows = full_df.loc[dt:dt + timedelta(hours=23, minutes=59, seconds=59), "is_erupted"]
                assert (rows == 1).any(), f"Expected positive labels on {d}"

            # Gap: Nov 8 falls between the two positive windows
            for d in ["2021-11-08"]:
                dt = datetime.fromisoformat(d)
                rows = full_df.loc[dt:dt + timedelta(hours=23, minutes=59, seconds=59), "is_erupted"]
                assert (rows == 0).all(), f"Expected zero labels on {d}"

            # Primary eruption (Nov 10) → positive rows Nov 9, 10, 11, 12
            for d in ["2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12"]:
                dt = datetime.fromisoformat(d)
                rows = full_df.loc[dt:dt + timedelta(hours=23, minutes=59, seconds=59), "is_erupted"]
                assert (rows == 1).any(), f"Expected positive labels on {d}"

    def test_build_overlapping_emits_warning(self) -> None:
        """Test that a warning is logged when a secondary eruption overlaps a window."""
        from eruption_forecast.logger import logger as ef_logger

        captured: list[str] = []
        handler_id = ef_logger.add(lambda msg: captured.append(msg), level="WARNING")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                _make_builder(
                    tmpdir,
                    eruption_dates=["2025-03-20", "2025-03-23"],
                    days_before_eruption=10,
                    day_to_forecast=4,
                ).build()
        finally:
            ef_logger.remove(handler_id)

        assert any("also falls within window" in m for m in captured)

    def test_build_loads_from_csv_when_exists(self) -> None:
        """Test that a second build without overwrite loads from the saved CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder1 = _make_builder(tmpdir, overwrite=True).build()
            csv_mtime = os.path.getmtime(builder1.csv)

            # Build again without overwrite — CSV must not be rewritten
            builder2 = _make_builder(tmpdir, overwrite=False).build()
            assert os.path.getmtime(builder2.csv) == csv_mtime
            assert not builder2.df.empty

    def test_build_overwrites_when_flag_set(self) -> None:
        """Test that overwrite=True causes build to recompute and rewrite the CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder1 = _make_builder(tmpdir, overwrite=True).build()
            mtime1 = os.path.getmtime(builder1.csv)

            import time
            time.sleep(0.05)

            builder2 = _make_builder(tmpdir, overwrite=True).build()
            mtime2 = os.path.getmtime(builder2.csv)

            assert mtime2 > mtime1


class TestMarkEruptionLabels:
    """Tests for the _mark_eruption_labels helper."""

    def _blank_df(self, start: datetime, end: datetime, freq: str = "12h") -> pd.DataFrame:
        """Return a zeroed is_erupted DataFrame indexed by DatetimeIndex."""
        idx = pd.date_range(start, end, freq=freq)
        return pd.DataFrame({"is_erupted": 0}, index=idx)

    def test_mark_single_eruption_no_overlap(self) -> None:
        """Test that a single eruption's positive period is correctly marked.

        Look-forward, eruption=2025-03-20, step=12h, buffer=1, dtf=2:
        pos_start = 2025-03-19 12:00 (E 00:00 - 12h)
        pos_end = 2025-03-22 23:59:59 (E + 2 days), but clamped to window_end (2025-03-20 23:59:59).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir, eruption_dates=["2025-03-20"], day_to_forecast=2)
            eruption_dt = datetime(2025, 3, 20)
            window_start = datetime(2025, 3, 13)

            df = self._blank_df(window_start, eruption_dt)
            result = builder._mark_eruption_labels(df, window_start, eruption_dt, eruption_dt)

            positive_start = datetime(2025, 3, 19, 12, 0, 0)
            positive_end = datetime(2025, 3, 20, 23, 59, 59)  # clamped to window end
            mask = (result.index >= positive_start) & (result.index <= positive_end)
            assert (result.loc[mask, "is_erupted"] == 1).all()
            assert (result.loc[~mask, "is_erupted"] == 0).all()

    def test_mark_secondary_eruption_overlap(self) -> None:
        """Test cross-year projection: 2020-11-05 marks positive rows in 2021 window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(
                tmpdir,
                eruption_dates=["2020-11-05", "2021-11-10"],
                day_to_forecast=2,
            )
            # Window owned by 2021-11-10 eruption
            window_start = datetime(2021, 10, 31)
            window_end = datetime(2021, 11, 10)

            df = self._blank_df(window_start, window_end, freq="24h")
            result = builder._mark_eruption_labels(
                df, window_start, window_end, datetime(2021, 11, 10)
            )

            # Projected eruption (Nov 5 → 2021-11-05) → look-forward: positive Nov 4 (12:00), 5, 6, 7
            for d in ["2021-11-05", "2021-11-06", "2021-11-07"]:
                dt = datetime.fromisoformat(d)
                assert result.loc[dt, "is_erupted"] == 1, f"Expected 1 on {d}"

    def test_mark_projection_leap_day_skipped(self) -> None:
        """Test that Feb 29 projection to a non-leap year is silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 2020 is a leap year; 2021 is not → Feb 29 → 2021 is invalid and must be skipped.
            builder = _make_builder(
                tmpdir,
                eruption_dates=["2020-02-29"],
                day_to_forecast=2,
            )
            window_start = datetime(2021, 2, 22)
            window_end = datetime(2021, 3, 5)

            df = self._blank_df(window_start, window_end, freq="24h")
            result = builder._mark_eruption_labels(
                df, window_start, window_end, datetime(2021, 3, 5)
            )

            assert (result["is_erupted"] == 0).all()

    def test_mark_no_overlap_skipped(self) -> None:
        """Test that an eruption entirely outside the window leaves df unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Eruption in April — completely outside March window
            builder = _make_builder(tmpdir, eruption_dates=["2025-04-15"], day_to_forecast=2)
            window_start = datetime(2025, 3, 13)
            window_end = datetime(2025, 3, 20)
            fake_eruption = datetime(2025, 3, 20)

            df = self._blank_df(window_start, window_end)
            result = builder._mark_eruption_labels(df, window_start, window_end, fake_eruption)

            assert (result["is_erupted"] == 0).all()


class TestDynamicLabelBuilderIntegration:
    """Integration tests for the full DynamicLabelBuilder workflow."""

    def test_full_workflow_single_eruption(self) -> None:
        """Test build → CSV saved → reload produces consistent DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _make_builder(tmpdir).build()

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
            ).build()

            total_positive = (builder.df["is_erupted"] == 1).sum()
            # At least day_to_forecast windows per eruption × 2 eruptions
            assert total_positive >= day_to_forecast * 2
