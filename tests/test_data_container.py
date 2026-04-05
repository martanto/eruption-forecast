"""Tests for BaseDataContainer, TremorData, and LabelData CSV-backed containers."""

import os
import tempfile

import pandas as pd
import pytest

from eruption_forecast.data_container import BaseDataContainer
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_data import LabelData


class TestBaseDataContainer:
    def test_is_abstract(self):
        """BaseDataContainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataContainer()

    def test_filename_property(self, tmp_path):
        """filename returns the basename including extension."""
        from eruption_forecast.label.label_data import LabelData

        fname = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
        path = tmp_path / fname
        df = pd.DataFrame(
            {"id": [0], "is_erupted": [0]},
            index=pd.to_datetime(["2020-06-01"]),
        )
        df.index.name = "datetime"
        df.to_csv(path)

        ld = LabelData(str(path))
        assert ld.filename == fname

    def test_basename_property(self, tmp_path):
        fname = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
        path = tmp_path / fname
        df = pd.DataFrame(
            {"id": [0], "is_erupted": [0]},
            index=pd.to_datetime(["2020-06-01"]),
        )
        df.index.name = "datetime"
        df.to_csv(path)

        ld = LabelData(str(path))
        assert ld.basename == "label_2020-01-01_2020-12-31_step-12-hours_dtf-2"

    def test_filetype_property(self, tmp_path):
        fname = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
        path = tmp_path / fname
        df = pd.DataFrame(
            {"id": [0], "is_erupted": [0]},
            index=pd.to_datetime(["2020-06-01"]),
        )
        df.index.name = "datetime"
        df.to_csv(path)

        ld = LabelData(str(path))
        assert ld.filetype == "csv"


class TestTremorData:
    def _make_tremor_df(self) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=144, freq="10min")
        return pd.DataFrame(
            {
                "rsam_f0": range(144),
                "rsam_f1": range(144, 288),
                "dsar_f0-f1": [0.5] * 144,
            },
            index=idx,
        )

    def test_init_with_df(self):
        td = TremorData(df=self._make_tremor_df())
        assert td.df is not None

    def test_start_end_dates(self):
        td = TremorData(df=self._make_tremor_df())
        assert td.start_date_str == "2025-01-01"
        assert td.end_date_str == "2025-01-01"

    def test_columns_property(self):
        td = TremorData(df=self._make_tremor_df())
        assert "rsam_f0" in td.columns

    def test_csv_default_empty(self):
        td = TremorData()
        assert td.csv == ""

    def test_has_csv_attribute(self):
        td = TremorData()
        assert hasattr(td, "csv")

    def test_from_csv(self, tmp_path):
        df = self._make_tremor_df()
        csv_path = str(tmp_path / "tremor.csv")
        df.index.name = "datetime"
        df.to_csv(csv_path)

        td = TremorData()
        loaded = td.from_csv(csv_path)
        assert isinstance(loaded, pd.DataFrame)
        assert "rsam_f0" in loaded.columns


class TestLabelData:
    def _make_label_csv(self, tmp_path) -> str:
        fname = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
        path = tmp_path / fname
        df = pd.DataFrame(
            {"id": [0, 1, 2], "is_erupted": [0, 0, 1]},
            index=pd.to_datetime(["2020-06-01", "2020-06-02", "2020-06-14"]),
        )
        df.index.name = "datetime"
        df.to_csv(path)
        return str(path)

    def test_parameters_parsed(self, tmp_path):
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        p = ld.parameters
        assert p["window_step"] == 12
        assert p["window_unit"] == "hours"
        assert p["day_to_forecast"] == 2

    def test_data_returns_dataframe(self, tmp_path):
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        assert isinstance(ld.data, pd.DataFrame)

    def test_no_kwargs_attribute(self, tmp_path):
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        assert not hasattr(ld, "kwargs")

    def test_start_end_date_strings(self, tmp_path):
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        assert ld.start_date_str == "2020-01-01"
        assert ld.end_date_str == "2020-12-31"
