"""Tests for the smaller helpers in ``utils.dataframe``, ``utils.date_utils``,
``utils.pathutils``, the ``@timer`` decorator, and the summary/explanation
dataclasses.
"""

import os
import importlib
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from loguru import logger

from eruption_forecast.utils.dataframe import (
    to_series,
    load_label_csv,
    get_envelope_values,
    load_datetime_indexed,
)
from eruption_forecast.utils.pathutils import (
    save_data,
    ensure_dir,
    load_pickle,
    save_figure,
    setup_nslc_directories,
    generate_features_filepaths,
)
from eruption_forecast.utils.date_utils import normalize_dates, to_datetime_index
from eruption_forecast.utils.validation import check_sampling_consistency
from eruption_forecast.dataclass.classifier_ensemble_summary import (
    SeedSummary,
    EruptionWindow,
    ProbabilityPick,
    ClassifierEnsembleSummary,
)


# ``eruption_forecast.decorators`` re-exports ``timer`` under the submodule's
# name, so reach the module itself through ``importlib``.
timer_module = importlib.import_module("eruption_forecast.decorators.timer")


# ---------------------------------------------------------------------------
# utils.dataframe
# ---------------------------------------------------------------------------


class TestToSeries:
    def test_builds_id_indexed_series(self) -> None:
        df = pd.DataFrame({"id": [3, 4], "is_erupted": [0, 1]})
        series = to_series(df, column_value="is_erupted")
        assert series.index.tolist() == [3, 4]
        assert series.tolist() == [0, 1]

    @pytest.mark.parametrize("kwargs", [{"column_value": "x"}, {"column_value": "is_erupted", "column_index": "x"}])
    def test_missing_column_raises(self, kwargs) -> None:
        with pytest.raises(ValueError):
            to_series(pd.DataFrame({"id": [1], "is_erupted": [0]}), **kwargs)


class TestLoadLabelCsv:
    def test_returns_id_indexed_labels(self, tmp_path) -> None:
        path = tmp_path / "label.csv"
        pd.DataFrame(
            {"datetime": ["2025-01-01", "2025-01-02"], "id": [0, 1], "is_erupted": [0, 1]}
        ).to_csv(path, index=False)
        labels = load_label_csv(str(path))
        assert labels.index.tolist() == [0, 1]
        assert labels.tolist() == [0, 1]


class TestLoadDatetimeIndexed:
    @pytest.fixture()
    def label_csv(self, tmp_path) -> str:
        index = pd.date_range("2025-01-01", periods=3, freq="6h", name="datetime")
        path = tmp_path / "label.csv"
        pd.DataFrame({"id": [0, 1, 2], "is_erupted": [0, 0, 1]}, index=index).to_csv(path)
        return str(path)

    @pytest.mark.parametrize("suffix", [".parquet", ".csv"])
    def test_attaches_datetime_index(self, label_csv, tmp_path, suffix) -> None:
        features = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3, name="id"))
        path = tmp_path / f"features{suffix}"
        if suffix == ".parquet":
            features.to_parquet(path)
        else:
            features.to_csv(path)

        result = load_datetime_indexed(label_csv, str(path))

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result["f"].tolist() == [1.0, 2.0, 3.0]
        assert str(result.index[1]) == "2025-01-01 06:00:00"

    def test_unsupported_suffix_raises(self, label_csv) -> None:
        with pytest.raises(ValueError):
            load_datetime_indexed(label_csv, "features.txt")


class TestGetEnvelopeValues:
    def test_adds_consensus_envelope_columns(self) -> None:
        df = pd.DataFrame(
            {
                "rf_eruption_probability": [0.1, 0.9, 0.2],
                "xgb_eruption_probability": [0.3, 0.5, 0.4],
                "rf_prediction": [0, 1, 0],
                "xgb_prediction": [0, 1, 1],
                "consensus_eruption_probability": [9.0, 9.0, 9.0],
            }
        )
        result = get_envelope_values(df)
        assert result["consensus_probability_max"].tolist() == [0.3, 0.9, 0.4]
        assert result["consensus_probability_min"].tolist() == [0.1, 0.5, 0.2]
        assert result["consensus_prediction_max"].tolist() == [0, 1, 1]
        # Rolling envelope must bound the raw extrema everywhere.
        assert (result["consensus_probability_max_envelope"] >= result["consensus_probability_max"]).all()
        assert (result["consensus_probability_min_envelope"] <= result["consensus_probability_min"]).all()

    def test_missing_probability_columns_raise(self) -> None:
        with pytest.raises(ValueError):
            get_envelope_values(pd.DataFrame({"rf_prediction": [0]}))

    def test_missing_prediction_columns_raise(self) -> None:
        with pytest.raises(ValueError):
            get_envelope_values(pd.DataFrame({"rf_eruption_probability": [0.1]}))


# ---------------------------------------------------------------------------
# utils.date_utils
# ---------------------------------------------------------------------------


def test_normalize_dates_spans_full_days() -> None:
    start, end, start_str, end_str = normalize_dates("2025-01-01", datetime(2025, 1, 3, 12))
    assert start == datetime(2025, 1, 1, 0, 0, 0)
    assert end == datetime(2025, 1, 3, 23, 59, 59)
    assert (start_str, end_str) == ("2025-01-01", "2025-01-03")


class TestToDatetimeIndex:
    @pytest.fixture()
    def datetime_map(self) -> pd.DataFrame:
        return pd.DataFrame({"datetime": ["2025-01-01", "2025-01-02"]}, index=[1, 2])

    def test_merges_on_index(self, datetime_map) -> None:
        df = pd.DataFrame({"feature_1": [0.5, 0.8]}, index=[1, 2])
        result = to_datetime_index(datetime_map, df)
        assert result.columns.tolist() == ["feature_1"]
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_merges_on_column(self, datetime_map) -> None:
        df = pd.DataFrame({"id": [1, 2], "feature_1": [0.5, 0.8]})
        result = to_datetime_index(datetime_map, df, on="id")
        assert result.columns.tolist() == ["feature_1"]
        assert str(result.index[1].date()) == "2025-01-02"

    def test_datetime_indexed_label_map(self) -> None:
        labels = pd.DataFrame(
            {"id": [0, 1]}, index=pd.DatetimeIndex(["2025-01-01", "2025-01-02"], name="datetime")
        )
        df = pd.DataFrame({"p": [0.1, 0.2]}, index=[0, 1])
        result = to_datetime_index(labels, df)
        assert result.index.tolist() == list(labels.index)

    def test_already_datetime_indexed_passthrough(self, datetime_map) -> None:
        df = pd.DataFrame({"x": [1]}, index=pd.DatetimeIndex(["2025-01-01"]))
        assert to_datetime_index(datetime_map, df) is df

    def test_length_mismatch_raises(self, datetime_map) -> None:
        with pytest.raises(ValueError):
            to_datetime_index(datetime_map, pd.DataFrame({"x": [1]}, index=[1]))

    def test_missing_on_column_raises(self, datetime_map) -> None:
        with pytest.raises(ValueError):
            to_datetime_index(datetime_map, pd.DataFrame({"x": [1, 2]}), on="id")

    def test_datetime_index_map_without_id_raises(self) -> None:
        labels = pd.DataFrame({"is_erupted": [0]}, index=pd.DatetimeIndex(["2025-01-01"]))
        with pytest.raises(ValueError):
            to_datetime_index(labels, pd.DataFrame({"x": [1]}))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "is_consistent is computed as inconsistent_data.empty, which is True "
        "for any frame without columns, even when rows are inconsistent."
    ),
)
def test_check_sampling_consistency_on_index_only_frame() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="6h")
    is_consistent, _, inconsistent, _ = check_sampling_consistency(
        pd.DataFrame(index=index), expected_freq="3h", tolerance="0s"
    )
    assert len(inconsistent) == 7
    assert is_consistent is False


# ---------------------------------------------------------------------------
# utils.pathutils
# ---------------------------------------------------------------------------


class TestLoadPickle:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "obj.pkl"
        joblib.dump({"a": 1}, path)
        assert load_pickle(str(path)) == {"a": 1}

    def test_missing_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pickle(str(tmp_path / "missing.pkl"))

    def test_truncated_raises_runtime_error(self, tmp_path) -> None:
        path = tmp_path / "obj.pkl"
        joblib.dump(list(range(1000)), path)
        path.write_bytes(path.read_bytes()[:20])
        with pytest.raises(RuntimeError):
            load_pickle(str(path))

    def test_garbage_bytes_raise_runtime_error(self, tmp_path) -> None:
        path = tmp_path / "obj.pkl"
        path.write_bytes(b"not a pickle")
        with pytest.raises(RuntimeError):
            load_pickle(str(path))


def test_ensure_dir_empty_string_is_noop() -> None:
    assert ensure_dir("") == ""


class TestSaveFigure:
    def test_appends_extension_and_closes(self, tmp_path) -> None:
        fig, _ = plt.subplots()
        save_figure(fig, str(tmp_path / "sub" / "plot"), dpi=50, verbose=False)
        assert (tmp_path / "sub" / "plot.png").is_file()
        assert not plt.fignum_exists(fig.number)

    def test_keeps_existing_extension_and_writes_pdf(self, tmp_path) -> None:
        fig, _ = plt.subplots()
        save_figure(fig, str(tmp_path / "plot.png"), dpi=50, save_as_pdf=True, verbose=False)
        assert (tmp_path / "plot.png").is_file()
        assert (tmp_path / "plot.pdf").is_file()


class TestSaveData:
    def test_csv(self, tmp_path) -> None:
        save_data(pd.DataFrame({"a": [1]}), str(tmp_path / "d"))
        assert (tmp_path / "d.csv").is_file()

    def test_parquet_drops_index(self, tmp_path) -> None:
        save_data(pd.DataFrame({"a": [1]}, index=["x"]), str(tmp_path / "d"), filetype="parquet")
        assert pd.read_parquet(tmp_path / "d.parquet").index.tolist() == [0]

    def test_other_types_use_joblib(self, tmp_path) -> None:
        save_data({"k": np.arange(3)}, str(tmp_path / "d"), filetype="pkl")
        assert joblib.load(tmp_path / "d.pkl")["k"].tolist() == [0, 1, 2]


def test_setup_nslc_directories(tmp_path) -> None:
    nslc, output_dir, station_dir = setup_nslc_directories(
        "vg", "ojn", "00", "ehz", output_dir=str(tmp_path)
    )
    assert nslc == "vg.ojn.00.ehz"
    assert output_dir == str(tmp_path)
    assert station_dir == os.path.join(str(tmp_path), "VG.OJN.00.EHZ")


class TestGenerateFeaturesFilepaths:
    def _call(self, tmp_path, **kwargs):
        return generate_features_filepaths(
            random_state=42,
            features_seed_dir=str(tmp_path / "seed"),
            features_resampled_dir=str(tmp_path / "resampled"),
            figures_seed_dir=str(tmp_path / "figures"),
            **kwargs,
        )

    def test_paths_and_no_skip_when_missing(self, tmp_path) -> None:
        can_skip, seed_path, resampled_path, figure_path = self._call(tmp_path)
        assert not can_skip
        assert seed_path.endswith(os.path.join("seed", "00042.csv"))
        assert resampled_path.endswith(os.path.join("resampled", "00042.csv"))
        assert figure_path is None

    def test_skip_when_outputs_exist(self, tmp_path) -> None:
        for sub in ("seed", "resampled"):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / "00042.csv").write_text("x")
        assert self._call(tmp_path)[0] is True
        assert self._call(tmp_path, overwrite=True)[0] is False
        # A requested-but-missing figure also forces a rebuild.
        assert self._call(tmp_path, plot_features=True)[0] is False


# ---------------------------------------------------------------------------
# @timer
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_messages():
    """Collect every loguru message emitted while the test runs."""
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield messages
    logger.remove(handler_id)


class TestTimer:
    def test_passthrough_and_logs(self, log_messages) -> None:
        @timer_module.timer()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert add.__name__ == "add"
        assert any(m.startswith("add: Took 00 hours") for m in log_messages)

    def test_custom_name(self, log_messages) -> None:
        @timer_module.timer(name="my task")
        def noop() -> None:
            return None

        noop()
        assert any(m.startswith("my task: Took") for m in log_messages)

    def test_telegram_forwarding(self, monkeypatch) -> None:
        sent: list[str] = []

        class FakeTelegram:
            def send_message(self, message: str):
                sent.append(message)
                return self

        monkeypatch.setattr(timer_module, "TelegramNotification", FakeTelegram)

        @timer_module.timer(name="job", send_to="telegram")
        def job() -> str:
            return "done"

        assert job() == "done"
        assert len(sent) == 1
        assert sent[0].startswith("job: Took")

    def test_exception_propagates_without_logging(self, log_messages) -> None:
        @timer_module.timer(name="boom")
        def boom() -> None:
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            boom()
        assert not any(m.startswith("boom: Took") for m in log_messages)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestSummaryDataclasses:
    @pytest.fixture()
    def pick(self) -> ProbabilityPick:
        return ProbabilityPick(
            random_state=1, index=2, datetime=pd.Timestamp("2025-01-01"), value=0.7
        )

    def test_picks_are_frozen(self, pick) -> None:
        with pytest.raises(AttributeError):
            pick.value = 0.1  # type: ignore[misc]

    def test_eruption_window_repr_counts_seeds(self, pick) -> None:
        window = EruptionWindow(
            eruption_date="2025-01-01",
            highest=pick,
            lowest=pick,
            seeds=[SeedSummary(random_state=1, highest=pick, lowest=pick)] * 3,
        )
        assert "seeds=3" in repr(window)

    def test_summary_defaults(self) -> None:
        summary = ClassifierEnsembleSummary(classifier_name="rf")
        assert summary.eruption_windows == []
        assert summary.highest is None
        assert summary.lowest is None
        assert "eruption_windows=0" in repr(summary)
