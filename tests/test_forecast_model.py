"""Unit tests for ForecastModel (Phase 4 refactoring).

Tests cover initialisation validation, the validate/create_directories
separation, and the predict() pipeline method.  All tests use synthetic
in-memory data and temporary directories — no real seismic data required.
"""

# Standard library imports
import os
import tempfile

# Third party imports
import pytest

# Project imports
from eruption_forecast.model.forecast_model import ForecastModel


# ---------------------------------------------------------------------------
# Helpers — default constructor kwargs
# ---------------------------------------------------------------------------


def _valid_kwargs(output_dir: str) -> dict:
    """Return a minimal set of valid kwargs for ForecastModel.__init__."""
    return {
        "station": "OJN",
        "channel": "EHZ",
        "window_size": 1,
        "volcano_id": "VOLCANO_001",
        "network": "VG",
        "location": "00",
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# ForecastModel — initialisation & validation
# ---------------------------------------------------------------------------


class TestForecastModelInit:
    """Test ForecastModel construction and validation."""

    def test_valid_initialization(self) -> None:
        """ForecastModel accepts valid parameters and sets attributes."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_valid_kwargs(tmp))
            assert fm.station == "OJN"
            assert fm.channel == "EHZ"
            assert fm.window_size == 1
            assert fm.volcano_id == "VOLCANO_001"
            assert fm.network == "VG"
            assert fm.location == "00"

    def test_empty_station_raises(self) -> None:
        """ValueError when station is an empty / whitespace string."""
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = _valid_kwargs(tmp)
            kwargs["station"] = "   "
            with pytest.raises(ValueError, match="station cannot be empty"):
                ForecastModel(**kwargs)

    def test_empty_channel_raises(self) -> None:
        """ValueError when channel is an empty / whitespace string."""
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = _valid_kwargs(tmp)
            kwargs["channel"] = ""
            with pytest.raises(ValueError, match="channel cannot be empty"):
                ForecastModel(**kwargs)

    def test_empty_volcano_id_raises(self) -> None:
        """ValueError when volcano_id is an empty / whitespace string."""
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = _valid_kwargs(tmp)
            kwargs["volcano_id"] = " "
            with pytest.raises(ValueError, match="volcano_id cannot be empty"):
                ForecastModel(**kwargs)

    def test_zero_window_size_raises(self) -> None:
        """ValueError when window_size is zero."""
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = _valid_kwargs(tmp)
            kwargs["window_size"] = 0
            with pytest.raises(ValueError, match="window_size must be greater than 0"):
                ForecastModel(**kwargs)

    def test_negative_n_jobs_raises(self) -> None:
        """ValueError when n_jobs is negative."""
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = _valid_kwargs(tmp)
            kwargs["n_jobs"] = -1
            with pytest.raises(ValueError, match="n_jobs must be greater than 0"):
                ForecastModel(**kwargs)

    def test_start_after_end_raises(self) -> None:
        """ValueError when start_date is after end_date (raised by calculate())."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_valid_kwargs(tmp))
            with pytest.raises(ValueError):
                fm.calculate(start_date="2020-06-01", end_date="2020-01-01")

    def test_output_directories_created(self) -> None:
        """All three output directories exist after init."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "output")
            kwargs = _valid_kwargs(out)
            fm = ForecastModel(**kwargs)
            assert os.path.isdir(fm.output_dir)
            assert os.path.isdir(fm.station_dir)
            assert os.path.isdir(fm.features_dir)


# ---------------------------------------------------------------------------
# ForecastModel — validate / create_directories separation
# ---------------------------------------------------------------------------


class TestForecastModelValidate:
    """Prove that validate() no longer creates directories."""

    def test_validate_does_not_create_dirs(self) -> None:
        """Calling validate() alone does not recreate deleted directories."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_valid_kwargs(tmp))

            # Directories exist after init (created by create_directories)
            assert os.path.isdir(fm.features_dir)

            # Remove them
            os.rmdir(fm.features_dir)
            os.rmdir(fm.station_dir)

            # validate() must NOT recreate them
            fm.validate()
            assert not os.path.isdir(fm.station_dir)
            assert not os.path.isdir(fm.features_dir)


# ---------------------------------------------------------------------------
# ForecastModel — method chaining
# ---------------------------------------------------------------------------


class TestForecastModelChaining:
    """Test that ForecastModel pipeline methods return self for chaining."""

    def test_load_tremor_data_returns_self(self) -> None:
        """load_tremor_data() returns self for method chaining."""
        import pandas as pd
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_valid_kwargs(tmp))
            # Inject a pre-built tremor df instead of calculating
            n = 144 * 7
            idx = pd.date_range("2020-01-01", periods=n, freq="10min", name="datetime")
            rng = np.random.default_rng(0)
            df = pd.DataFrame(
                {"rsam_f0": rng.uniform(0, 1, n), "rsam_f1": rng.uniform(0, 1, n)},
                index=idx,
            )
            csv = os.path.join(tmp, "tremor.csv")
            df.to_csv(csv)

            result = fm.load_tremor_data(csv)
            assert result is fm

    def test_set_feature_selection_method_returns_self(self) -> None:
        """set_feature_selection_method() returns self for method chaining."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_valid_kwargs(tmp))
            result = fm.set_feature_selection_method("tsfresh")
            assert result is fm
