"""Tests for TremorMatrixBuilder."""

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


def _make_tremor_df(start="2025-01-01", days=5) -> pd.DataFrame:
    """Create a synthetic tremor DataFrame at 10-minute intervals.

    Uses named 'datetime' index so TremorMatrixBuilder can reset_index correctly.
    """
    n = days * 144  # 144 samples/day at 10min
    idx = pd.date_range(start, periods=n, freq="10min", name="datetime")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rsam_f0": rng.uniform(0.1, 1.0, n),
            "rsam_f1": rng.uniform(0.1, 1.0, n),
            "dsar_f0-f1": rng.uniform(0.3, 0.7, n),
        },
        index=idx,
    )


def _make_label_df() -> pd.DataFrame:
    """Create a synthetic label DataFrame with window IDs.

    Labels start at 2025-01-03 so each window has a full preceding day of tremor
    (window_size=1 means the window covers [label_time - 1 day, label_time - 1ms]).
    """
    idx = pd.date_range("2025-01-03", periods=3, freq="12h", name="datetime")
    return pd.DataFrame(
        {
            "id": range(3),
            "is_erupted": [0, 0, 1],
        },
        index=idx,
    )


class TestTremorMatrixBuilderInit:
    def test_basic_init(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        )
        assert builder.window_size == 1

    def test_non_datetime_index_raises(self, tmp_path):
        tremor_df = _make_tremor_df().reset_index(drop=True)  # integer index
        label_df = _make_label_df()
        with pytest.raises(TypeError):
            TremorMatrixBuilder(
                tremor_df=tremor_df,
                label_df=label_df,
                output_dir=str(tmp_path),
            )

    def test_missing_id_column_raises(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df().drop(columns=["id"])
        with pytest.raises(ValueError):
            TremorMatrixBuilder(
                tremor_df=tremor_df,
                label_df=label_df,
                output_dir=str(tmp_path),
            )


class TestTremorMatrixBuilderBuild:
    def test_build_returns_self(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        )
        result = builder.build()
        assert result is builder

    def test_df_not_empty(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        ).build()
        assert builder.df is not None
        assert len(builder.df) > 0

    def test_id_column_present(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        ).build()
        assert "id" in builder.df.columns

    def test_select_tremor_columns(self, tmp_path):
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        ).build(select_tremor_columns=["rsam_f0"])
        # Only selected column should be in matrix (plus id/datetime)
        assert "rsam_f0" in builder.df.columns
        assert "rsam_f1" not in builder.df.columns

    def test_saves_csv_file(self, tmp_path):
        import os
        tremor_df = _make_tremor_df()
        label_df = _make_label_df()
        builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=label_df,
            output_dir=str(tmp_path),
            window_size=1,
        ).build()
        assert builder.csv is not None
        assert os.path.isfile(builder.csv)
