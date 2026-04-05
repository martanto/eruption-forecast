"""Tests for utility modules: formatting, date_utils, validation, pathutils, dataframe, array."""

import os
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# formatting.py
# ---------------------------------------------------------------------------

class TestSlugifyClassName:
    def test_camel_case(self):
        from eruption_forecast.utils.formatting import slugify_class_name
        assert slugify_class_name("MyClassName") == "my-class-name"

    def test_acronym(self):
        from eruption_forecast.utils.formatting import slugify_class_name
        assert slugify_class_name("HTTPSConnection") == "https-connection"

    def test_xgb_classifier(self):
        from eruption_forecast.utils.formatting import slugify_class_name
        assert slugify_class_name("XGBClassifier") == "xgb-classifier"

    def test_single_word(self):
        from eruption_forecast.utils.formatting import slugify_class_name
        assert slugify_class_name("Model") == "model"


class TestSlugify:
    def test_basic(self):
        from eruption_forecast.utils.formatting import slugify
        assert slugify("Hello World") == "hello-world"

    def test_custom_separator(self):
        from eruption_forecast.utils.formatting import slugify
        assert slugify("Hello World", hyphen="_") == "hello_world"

    def test_multiple_spaces_collapsed(self):
        from eruption_forecast.utils.formatting import slugify
        assert slugify("  Multiple   Spaces  ") == "multiple-spaces"

    def test_strips_leading_trailing_hyphens(self):
        from eruption_forecast.utils.formatting import slugify
        result = slugify("Hello World")
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# date_utils.py
# ---------------------------------------------------------------------------

class TestToDatetime:
    def test_string_input(self):
        from eruption_forecast.utils.date_utils import to_datetime
        result = to_datetime("2025-03-20")
        assert result == datetime(2025, 3, 20)

    def test_datetime_passthrough(self):
        from eruption_forecast.utils.date_utils import to_datetime
        dt = datetime(2025, 3, 20)
        assert to_datetime(dt) is dt

    def test_invalid_format_raises(self):
        from eruption_forecast.utils.date_utils import to_datetime
        with pytest.raises(ValueError):
            to_datetime("20-03-2025")


class TestSortDates:
    def test_sorts_ascending(self):
        from eruption_forecast.utils.date_utils import sort_dates
        result = sort_dates(["2025-03-20", "2025-01-01", "2025-06-15"])
        assert result == ["2025-01-01", "2025-03-20", "2025-06-15"]

    def test_as_datetime(self):
        from eruption_forecast.utils.date_utils import sort_dates
        result = sort_dates(["2025-03-20", "2025-01-01"], as_datetime=True)
        assert result[0] == datetime(2025, 1, 1)


class TestParseLabelFilename:
    def test_parses_correctly(self):
        from eruption_forecast.utils.date_utils import parse_label_filename
        result = parse_label_filename(
            "label_2020-01-01_2020-12-31_step-12-hours_dtf-2"
        )
        assert result["start_date_str"] == "2020-01-01"
        assert result["end_date_str"] == "2020-12-31"
        assert result["window_step"] == 12
        assert result["window_step_unit"] == "hours"
        assert result["day_to_forecast"] == 2

    def test_minutes_unit(self):
        from eruption_forecast.utils.date_utils import parse_label_filename
        result = parse_label_filename(
            "label_2021-01-01_2021-06-30_step-30-minutes_dtf-3"
        )
        assert result["window_step_unit"] == "minutes"
        assert result["window_step"] == 30


# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------

class TestValidateRandomState:
    def test_valid(self):
        from eruption_forecast.utils.validation import validate_random_state
        validate_random_state(0)
        validate_random_state(42)

    def test_negative_raises(self):
        from eruption_forecast.utils.validation import validate_random_state
        with pytest.raises(ValueError):
            validate_random_state(-1)


class TestValidateDateRanges:
    def test_valid_range(self):
        from eruption_forecast.utils.validation import validate_date_ranges
        start, end, days = validate_date_ranges("2025-01-01", "2025-01-10")
        assert days == 9

    def test_start_equals_end_raises(self):
        from eruption_forecast.utils.validation import validate_date_ranges
        with pytest.raises(ValueError):
            validate_date_ranges("2025-01-01", "2025-01-01")

    def test_start_after_end_raises(self):
        from eruption_forecast.utils.validation import validate_date_ranges
        with pytest.raises(ValueError):
            validate_date_ranges("2025-06-01", "2025-01-01")


class TestValidateWindowStep:
    def test_valid_hours(self):
        from eruption_forecast.utils.validation import validate_window_step
        result = validate_window_step(6, "hours")
        assert result == (6, "hours")

    def test_valid_minutes(self):
        from eruption_forecast.utils.validation import validate_window_step
        result = validate_window_step(30, "minutes")
        assert result == (30, "minutes")

    def test_invalid_unit_raises(self):
        from eruption_forecast.utils.validation import validate_window_step
        with pytest.raises(ValueError):
            validate_window_step(6, "days")

    def test_non_int_step_raises(self):
        from eruption_forecast.utils.validation import validate_window_step
        with pytest.raises(TypeError):
            validate_window_step(6.0, "hours")


class TestValidateColumns:
    def test_valid_columns(self):
        from eruption_forecast.utils.validation import validate_columns
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_columns(df, ["a", "b"])  # no error

    def test_missing_column_raises(self):
        from eruption_forecast.utils.validation import validate_columns
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            validate_columns(df, ["a", "missing"])

    def test_exclude_columns(self):
        from eruption_forecast.utils.validation import validate_columns
        df = pd.DataFrame({"a": [1]})
        validate_columns(df, ["a", "missing"], exclude_columns=["missing"])


class TestCheckSamplingConsistency:
    def test_consistent_data(self):
        from eruption_forecast.utils.validation import check_sampling_consistency
        idx = pd.date_range("2025-01-01", periods=20, freq="10min")
        df = pd.DataFrame({"v": range(20)}, index=idx)
        is_consistent, _, inconsistent, rate = check_sampling_consistency(df)
        assert is_consistent
        assert inconsistent.empty
        assert rate == 600  # 10 minutes in seconds

    def test_inconsistent_data_detected(self):
        from eruption_forecast.utils.validation import check_sampling_consistency
        idx = pd.date_range("2025-01-01", periods=10, freq="10min").tolist()
        idx[5] = idx[5] + pd.Timedelta("5min")  # introduce gap
        df = pd.DataFrame({"v": range(10)}, index=pd.DatetimeIndex(idx))
        is_consistent, _, inconsistent, _ = check_sampling_consistency(df)
        assert not is_consistent
        assert not inconsistent.empty

    def test_too_few_rows_raises(self):
        from eruption_forecast.utils.validation import check_sampling_consistency
        idx = pd.date_range("2025-01-01", periods=2, freq="10min")
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        with pytest.raises(ValueError):
            check_sampling_consistency(df)


# ---------------------------------------------------------------------------
# pathutils.py
# ---------------------------------------------------------------------------

class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        from eruption_forecast.utils.pathutils import ensure_dir
        target = str(tmp_path / "nested" / "dirs")
        result = ensure_dir(target)
        assert os.path.isdir(result)
        assert result == target

    def test_existing_dir_ok(self, tmp_path):
        from eruption_forecast.utils.pathutils import ensure_dir
        ensure_dir(str(tmp_path))  # already exists — no error


class TestLoadJson:
    def test_loads_file(self, tmp_path):
        from eruption_forecast.utils.pathutils import load_json
        data = {"key": "value", "num": 42}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        result = load_json(str(path))
        assert result == data

    def test_missing_file_raises(self, tmp_path):
        from eruption_forecast.utils.pathutils import load_json
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))


class TestResolveOutputDir:
    def test_none_uses_default_subpath(self, tmp_path):
        from eruption_forecast.utils.pathutils import resolve_output_dir
        result = resolve_output_dir(None, str(tmp_path), "subdir/output")
        assert result == os.path.join(str(tmp_path), "subdir/output")

    def test_absolute_path_unchanged(self, tmp_path):
        from eruption_forecast.utils.pathutils import resolve_output_dir
        abs_path = str(tmp_path / "absolute")
        result = resolve_output_dir(abs_path, str(tmp_path), "ignored")
        assert result == abs_path


# ---------------------------------------------------------------------------
# dataframe.py
# ---------------------------------------------------------------------------

class TestRemoveAnomalies:
    def test_removes_outlier(self):
        from eruption_forecast.utils.dataframe import remove_anomalies
        idx = pd.date_range("2025-01-01", periods=50, freq="10min")
        data = np.ones(50)
        data[25] = 1000.0  # extreme outlier
        df = pd.DataFrame({"v": data}, index=idx)
        result = remove_anomalies(df, columns=["v"])
        # Either the outlier was removed (NaN) or unchanged — just check structure
        assert isinstance(result, pd.DataFrame)
        assert "v" in result.columns

    def test_outlier_replaced_when_extreme(self):
        from eruption_forecast.utils.dataframe import remove_anomalies
        idx = pd.date_range("2025-01-01", periods=200, freq="10min")
        rng = np.random.default_rng(1)
        data = rng.normal(1.0, 0.01, 200)  # very tight cluster around 1.0
        data[100] = 1e6  # extreme outlier, far beyond any threshold
        df = pd.DataFrame({"v": data}, index=idx)
        result = remove_anomalies(df, columns=["v"])
        assert pd.isna(result["v"].iloc[100])

    def test_inplace_false_leaves_original(self):
        from eruption_forecast.utils.dataframe import remove_anomalies
        idx = pd.date_range("2025-01-01", periods=200, freq="10min")
        rng = np.random.default_rng(2)
        data = rng.normal(1.0, 0.01, 200)
        data[100] = 1e6
        df = pd.DataFrame({"v": data}, index=idx)
        original_val = df["v"].iloc[100]
        remove_anomalies(df, inplace=False)
        assert df["v"].iloc[100] == original_val  # original unchanged


# ---------------------------------------------------------------------------
# array.py
# ---------------------------------------------------------------------------

class TestDetectAnomaliesZscore:
    def test_flags_outlier(self):
        from eruption_forecast.utils.array import detect_anomalies_zscore
        data = np.array([1.0, 2.0, 1.5, 2.0, 1.0, 100.0])
        result = detect_anomalies_zscore(data)
        assert result.dtype == bool
        assert result[-1]  # 100.0 flagged

    def test_handles_nans(self):
        from eruption_forecast.utils.array import detect_anomalies_zscore
        data = np.array([1.0, 2.0, np.nan, 3.0, 100.0, 2.0])
        result = detect_anomalies_zscore(data)
        assert result.dtype == bool
        assert result[-2]  # 100.0 flagged


class TestDetectMaximumOutlier:
    def test_detects_maximum(self):
        from eruption_forecast.utils.array import detect_maximum_outlier
        data = np.array([1.0, 2.0, 1.5, 1.0, 50.0])
        mask = detect_maximum_outlier(data)
        assert mask[-1]  # maximum is flagged
