"""Unit tests for the feature extraction and model training modules.

Tests for FeaturesBuilder and TrainModel classes to verify correctness
after Phase 3 refactoring. All tests use synthetic in-memory data.
"""

# Standard library imports
import os
import tempfile

# Third party imports
import pandas as pd
import pytest

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ERUPTED_COLUMN,
    ID_COLUMN,
    REQUIRED_LABEL_COLUMNS,
    SECONDS_PER_DAY,
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.features.features_builder import FeaturesBuilder

# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------


def _make_tremor_df(days: int = 5, freq: str = "10min") -> pd.DataFrame:
    """Create a synthetic tremor DataFrame with DatetimeIndex named 'datetime'."""
    index = pd.date_range("2020-01-01", periods=days * 144, freq=freq, name="datetime")
    return pd.DataFrame(
        {"rsam_f0": range(len(index)), "dsar_f0-f1": range(len(index))},
        index=index,
    )


def _make_label_df(n: int = 3, start: str = "2020-01-03") -> pd.DataFrame:
    """Create a synthetic label DataFrame with DatetimeIndex.

    Labels start at *start* so that a 1-day lookback window falls within
    the tremor data produced by _make_tremor_df(days=5).
    """
    index = pd.date_range(start, periods=n, freq="12h")
    return pd.DataFrame(
        {ID_COLUMN: range(1, n + 1), ERUPTED_COLUMN: [0] * (n - 1) + [1]},
        index=index,
    )


# ---------------------------------------------------------------------------
# Constants module
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants module values."""

    def test_required_label_columns(self) -> None:
        assert REQUIRED_LABEL_COLUMNS == ["id", "is_erupted"]

    def test_seconds_per_day(self) -> None:
        assert SECONDS_PER_DAY == 86400

    def test_significant_features_filename(self) -> None:
        assert SIGNIFICANT_FEATURES_FILENAME == "significant_features.csv"

    def test_column_name_constants(self) -> None:
        assert ID_COLUMN == "id"
        assert DATETIME_COLUMN == "datetime"
        assert ERUPTED_COLUMN == "is_erupted"


# ---------------------------------------------------------------------------
# FeaturesBuilder — initialisation & validation
# ---------------------------------------------------------------------------


class TestFeaturesBuilderInit:
    """Test FeaturesBuilder construction and validation."""

    def test_valid_initialization(self) -> None:
        """FeaturesBuilder accepts valid tremor + label data."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(),
                df_label=_make_label_df(),
                output_dir=tmp,
                window_size=1,
            )
            assert fb.window_size == 1
            assert fb.output_dir == tmp

    def test_non_datetime_tremor_index_raises_type_error(self) -> None:
        """TypeError when df_tremor index is not DatetimeIndex."""
        df_bad = pd.DataFrame({"rsam_f0": [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(TypeError, match="df_tremor.index is not a DatetimeIndex"):
            FeaturesBuilder(
                df_tremor=df_bad,
                df_label=_make_label_df(),
                output_dir="/tmp/test",
                window_size=1,
            )

    def test_non_datetime_label_index_raises_type_error(self) -> None:
        """TypeError when df_label index is not DatetimeIndex."""
        df_bad = pd.DataFrame({ID_COLUMN: [1], ERUPTED_COLUMN: [0]}, index=[0])
        with pytest.raises(TypeError, match="df_label.index is not a DatetimeIndex"):
            FeaturesBuilder(
                df_tremor=_make_tremor_df(),
                df_label=df_bad,
                output_dir="/tmp/test",
                window_size=1,
            )

    def test_missing_id_column_raises_value_error(self) -> None:
        """ValueError when label DataFrame is missing 'id' column."""
        label = _make_label_df()
        label = label.drop(columns=[ID_COLUMN])
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Column 'id' not found"):
                FeaturesBuilder(
                    df_tremor=_make_tremor_df(),
                    df_label=label,
                    output_dir=tmp,
                    window_size=1,
                )

    def test_missing_is_erupted_column_raises_value_error(self) -> None:
        """ValueError when label DataFrame is missing 'is_erupted' column."""
        label = _make_label_df()
        label = label.drop(columns=[ERUPTED_COLUMN])
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Column 'is_erupted' not found"):
                FeaturesBuilder(
                    df_tremor=_make_tremor_df(),
                    df_label=label,
                    output_dir=tmp,
                    window_size=1,
                )

    def test_invalid_tremor_column_raises_value_error(self) -> None:
        """ValueError when a requested tremor column does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="not found in Tremor dataframe"):
                FeaturesBuilder(
                    df_tremor=_make_tremor_df(),
                    df_label=_make_label_df(),
                    output_dir=tmp,
                    window_size=1,
                    tremor_columns=["nonexistent_column"],
                )

    def test_tremor_column_filtering(self) -> None:
        """Only the requested tremor columns are kept after init."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(),
                df_label=_make_label_df(),
                output_dir=tmp,
                window_size=1,
                tremor_columns=["rsam_f0"],
            )
            assert fb.df_tremor.columns.tolist() == ["rsam_f0"]

    def test_output_directory_created(self) -> None:
        """create_directories() produces the output dir."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "output")
            FeaturesBuilder(
                df_tremor=_make_tremor_df(),
                df_label=_make_label_df(),
                output_dir=out,
                window_size=1,
            )
            assert os.path.isdir(out)


# ---------------------------------------------------------------------------
# FeaturesBuilder — build
# ---------------------------------------------------------------------------


class TestFeaturesBuilderBuild:
    """Test FeaturesBuilder.build() behaviour."""

    def test_build_produces_non_empty_matrix(self) -> None:
        """build() returns a non-empty DataFrame when data aligns."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
            )
            result = fb.build(save_per_method=False)
            assert not result.empty
            assert ID_COLUMN in result.columns

    def test_build_saves_csv(self) -> None:
        """build() writes the features CSV to output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
            )
            fb.build(save_per_method=False)
            assert os.path.isfile(os.path.join(tmp, "features.csv"))

    def test_build_custom_filename(self) -> None:
        """build() respects a custom filename parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
            )
            fb.build(save_per_method=False, filename="custom.csv")
            assert os.path.isfile(os.path.join(tmp, "custom.csv"))

    def test_build_skips_when_file_exists_and_no_overwrite(self) -> None:
        """build() returns cached CSV when file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
                overwrite=False,
            )
            first = fb.build(save_per_method=False)
            # Second call should load from disk without re-computing
            second = fb.build(save_per_method=False)
            assert first.shape == second.shape

    def test_build_overwrites_when_flag_set(self) -> None:
        """build() re-computes when overwrite=True."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
                overwrite=True,
            )
            first = fb.build(save_per_method=False)
            second = fb.build(save_per_method=False)
            assert first.shape == second.shape

    def test_build_raises_when_no_windows_match(self) -> None:
        """ValueError when no tremor windows align with labels."""
        # Labels far in the future — no tremor data to slice
        label = pd.DataFrame(
            {ID_COLUMN: [1], ERUPTED_COLUMN: [0]},
            index=pd.date_range("2025-06-01", periods=1, freq="12h"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=3),
                df_label=label,
                output_dir=tmp,
                window_size=1,
            )
            with pytest.raises(ValueError, match="Features matrix is empty"):
                fb.build(save_per_method=False)

    def test_build_raises_value_error_not_exception(self) -> None:
        """Ensure the empty-matrix error is ValueError, not bare Exception."""
        label = pd.DataFrame(
            {ID_COLUMN: [1], ERUPTED_COLUMN: [0]},
            index=pd.date_range("2025-06-01", periods=1, freq="12h"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=3),
                df_label=label,
                output_dir=tmp,
                window_size=1,
            )
            # Must be ValueError specifically — not a generic Exception
            with pytest.raises(ValueError):
                fb.build(save_per_method=False)


# ---------------------------------------------------------------------------
# FeaturesBuilder — save_features_per_method
# ---------------------------------------------------------------------------


class TestFeaturesBuilderSavePerMethod:
    """Test save_features_per_method behaviour."""

    def test_save_per_method_creates_files(self) -> None:
        """One CSV is created per tremor metric column."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
            )
            fb.build(save_per_method=True)
            method_dir = os.path.join(tmp, "method")
            assert os.path.isdir(method_dir)
            # Should have one file per tremor column (rsam_f0, dsar_f0-f1)
            metric_files = [f for f in os.listdir(method_dir) if f.endswith(".csv")]
            assert len(metric_files) == 2

    def test_save_per_method_skip_existing(self) -> None:
        """Second call does not overwrite existing per-method files."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                df_tremor=_make_tremor_df(days=5),
                df_label=_make_label_df(n=3, start="2020-01-03"),
                output_dir=tmp,
                window_size=1,
                overwrite=False,
            )
            result = fb.build(save_per_method=True)
            method_dir = os.path.join(tmp, "method")
            # Record mtime of first file
            files = sorted(os.listdir(method_dir))
            mtime_before = os.path.getmtime(os.path.join(method_dir, files[0]))

            # Call save_per_method again — files should not be touched
            fb.save_features_per_method(result)
            mtime_after = os.path.getmtime(os.path.join(method_dir, files[0]))
            assert mtime_before == mtime_after


# ---------------------------------------------------------------------------
# TrainModel — validation (uses CSV round-trip via synthetic data)
# ---------------------------------------------------------------------------


def _write_synthetic_csvs(
    tmp: str, n_features: int = 10, n_samples: int = 20
) -> tuple[str, str]:
    """Write synthetic features and label CSVs for TrainModel tests.

    Returns (features_csv, label_csv) paths.
    """
    # Third party imports
    import numpy as np

    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.random((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    features.index.name = ID_COLUMN
    features_csv = os.path.join(tmp, "features.csv")
    features.to_csv(features_csv)

    labels = pd.DataFrame(
        {
            ID_COLUMN: range(n_samples),
            ERUPTED_COLUMN: [0] * (n_samples - 4) + [1] * 4,
        }
    )
    label_csv = os.path.join(tmp, "labels.csv")
    labels.to_csv(label_csv, index=False)

    return features_csv, label_csv


class TestTrainModelValidation:
    """Test TrainModel validation logic."""

    def test_valid_initialization(self) -> None:
        """TrainModel initialises without error on valid data."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            model = TrainModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=os.path.join(tmp, "out"),
            )
            assert model.df_features.shape[0] == 20

    def test_empty_features_raises_value_error(self) -> None:
        """ValueError when features CSV has zero rows."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            # Empty features
            pd.DataFrame(columns=["feat_0"]).to_csv(os.path.join(tmp, "features.csv"))
            # Non-empty labels (won't matter — features checked first)
            pd.DataFrame({ID_COLUMN: [0], ERUPTED_COLUMN: [0]}).to_csv(
                os.path.join(tmp, "labels.csv"), index=False
            )

            with pytest.raises(ValueError, match="Features cannot be empty"):
                TrainModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_empty_labels_raises_value_error(self) -> None:
        """ValueError when label CSV has zero rows."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            # 5-row features
            pd.DataFrame({"feat_0": range(5)}).to_csv(os.path.join(tmp, "features.csv"))
            # Empty labels — only header
            pd.DataFrame(columns=[ID_COLUMN, ERUPTED_COLUMN]).to_csv(
                os.path.join(tmp, "labels.csv"), index=False
            )

            with pytest.raises(ValueError, match="Labels cannot be empty"):
                TrainModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """ValueError when features and labels row counts differ."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"feat_0": range(10)}).to_csv(
                os.path.join(tmp, "features.csv")
            )
            pd.DataFrame({ID_COLUMN: range(5), ERUPTED_COLUMN: [0] * 5}).to_csv(
                os.path.join(tmp, "labels.csv"), index=False
            )

            with pytest.raises(ValueError, match="do not match"):
                TrainModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_output_directories_created(self) -> None:
        """create_directories() produces expected subdirectories."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "predictions")
            model = TrainModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            assert os.path.isdir(out)
            assert os.path.isdir(model.significant_features_dir)

    def test_errors_are_value_error_not_assertion_error(self) -> None:
        """Confirm validation raises ValueError, not AssertionError."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"feat_0": range(3)}).to_csv(os.path.join(tmp, "features.csv"))
            pd.DataFrame({ID_COLUMN: range(7), ERUPTED_COLUMN: [0] * 7}).to_csv(
                os.path.join(tmp, "labels.csv"), index=False
            )

            # Must be ValueError — never AssertionError
            with pytest.raises(ValueError):
                TrainModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )


# ---------------------------------------------------------------------------
# Integration — FeaturesBuilder → TrainModel round-trip
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: build features then hand off to TrainModel init."""

    def test_features_builder_output_loads_in_train_model(self) -> None:
        """CSV produced by FeaturesBuilder can be consumed by TrainModel."""
        # Project imports
        from eruption_forecast.model.train_model import TrainModel

        with tempfile.TemporaryDirectory() as tmp:
            # 1. Build features
            tremor = _make_tremor_df(days=5)
            label = _make_label_df(n=3, start="2020-01-03")
            fb = FeaturesBuilder(
                df_tremor=tremor,
                df_label=label,
                output_dir=tmp,
                window_size=1,
            )
            features_df = fb.build(save_per_method=False)
            features_csv = os.path.join(tmp, "features.csv")
            assert os.path.isfile(features_csv)

            # 2. Build a matching label CSV with same row count
            n_rows = len(features_df)
            labels = pd.DataFrame(
                {
                    ID_COLUMN: range(n_rows),
                    ERUPTED_COLUMN: [0] * (n_rows - 1) + [1],
                }
            )
            label_csv = os.path.join(tmp, "labels.csv")
            labels.to_csv(label_csv, index=False)

            # 3. TrainModel should initialise successfully
            model = TrainModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=os.path.join(tmp, "predictions"),
            )
            assert model.df_features.shape[0] == n_rows
            assert model.df_labels.shape[0] == n_rows
