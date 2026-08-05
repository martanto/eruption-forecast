"""Unit tests for the feature extraction module.

Tests for the ``FeaturesBuilder`` class. All tests use synthetic in-memory data.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
    SECONDS_PER_DAY,
)
from eruption_forecast.features.features_builder import FeaturesBuilder


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------


def _make_tremor_matrix(n_windows: int = 4, samples_per_window: int = 144) -> pd.DataFrame:
    """Create a synthetic tremor matrix as produced by TremorMatrixBuilder.

    Each window has `samples_per_window` rows and two tremor columns.
    """
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2025-01-02")
    for win_id in range(n_windows):
        for s in range(samples_per_window):
            rows.append(
                {
                    ID_COLUMN: win_id,
                    DATETIME_COLUMN: base + pd.Timedelta(minutes=10 * s),
                    "rsam_f0": rng.uniform(0, 1),
                    "rsam_f1": rng.uniform(0, 1),
                }
            )
    return pd.DataFrame(rows)


def _make_label_df(n_windows: int = 4) -> pd.DataFrame:
    """Create a synthetic label DataFrame with DatetimeIndex and id/is_erupted."""
    idx = pd.date_range("2025-01-03", periods=n_windows, freq="12h")
    return pd.DataFrame(
        {
            ID_COLUMN: range(n_windows),
            ERUPTED_COLUMN: [0] * (n_windows - 1) + [1],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Constants module
# ---------------------------------------------------------------------------


class TestConstants:
    def test_seconds_per_day(self) -> None:
        assert SECONDS_PER_DAY == 86400

    def test_column_name_constants(self) -> None:
        assert ID_COLUMN == "id"
        assert DATETIME_COLUMN == "datetime"
        assert ERUPTED_COLUMN == "is_erupted"


# ---------------------------------------------------------------------------
# FeaturesBuilder — initialisation & validation
# ---------------------------------------------------------------------------


class TestFeaturesBuilderInit:
    def test_valid_initialization(self) -> None:
        """FeaturesBuilder accepts a valid tremor matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                tremor_matrix_df=_make_tremor_matrix(),
                output_dir=tmp,
            )
            assert fb.output_dir == tmp

    def test_missing_id_column_raises(self) -> None:
        """ValueError when tremor_matrix_df is missing 'id' column."""
        matrix = _make_tremor_matrix().drop(columns=[ID_COLUMN])
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises((ValueError, KeyError)):
                FeaturesBuilder(tremor_matrix_df=matrix, output_dir=tmp)

    def test_output_directory_stored(self) -> None:
        """FeaturesBuilder stores the output_dir attribute."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "features")
            fb = FeaturesBuilder(tremor_matrix_df=_make_tremor_matrix(), output_dir=out)
            assert fb.output_dir == out


# ---------------------------------------------------------------------------
# FeaturesBuilder — extract_features
# ---------------------------------------------------------------------------


class TestFeaturesBuilderExtract:
    def test_extract_returns_dataframe(self) -> None:
        """extract_features() returns a non-empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                tremor_matrix_df=_make_tremor_matrix(n_windows=4),
                output_dir=tmp,
                label_df=_make_label_df(n_windows=4),
            )
            result = fb.extract_features(select_tremor_columns=["rsam_f0", "rsam_f1"])
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_extract_selects_columns(self) -> None:
        """extract_features() only uses the specified tremor columns."""
        with tempfile.TemporaryDirectory() as tmp:
            fb = FeaturesBuilder(
                tremor_matrix_df=_make_tremor_matrix(n_windows=4),
                output_dir=tmp,
                label_df=_make_label_df(n_windows=4),
            )
            result = fb.extract_features(select_tremor_columns=["rsam_f0", "rsam_f1"])
            # All tsfresh feature columns contain the source column name
            tremor_cols = [c for c in result.columns if "__" in c]
            assert all("rsam_f0" in c or "rsam_f1" in c for c in tremor_cols)


