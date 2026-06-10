"""Unit tests for ForecastConfig and the new ForecastModel config persistence.

Covers the new-API counterpart of ``test_pipeline_config.py``:

- Each section dataclass: default values, ``to_dict()`` / ``from_dict()``
- ``ForecastConfig``: ``to_dict()``, ``save()``/``load()`` round-trips (YAML + JSON)
- ``ForecastConfig.load()`` with partial configs (missing sections)
- Unknown keys in ``from_dict()`` are silently ignored
- ``ForecastModel._config`` populated at ``__init__`` (model section only)
- ``ForecastModel.save_config()`` default path lives under
  ``{station_dir}/config/`` (sibling of the per-stage ``cache/`` directories
  written by :class:`~eruption_forecast.model.cache_model.CacheModel`)
- ``ForecastModel.from_config()`` round-trip
- ``ForecastModel.run()`` is a no-op when no stage sections are populated
- ``ForecastPredictConfig`` does not capture the variadic ``**plot_kwargs``
- ``ForecastConfig.load()`` raises ``FileNotFoundError`` for missing paths

All tests use temporary directories and in-memory data — no seismic files
required.
"""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import fields

import pytest

from eruption_forecast.model.forecast import ForecastModel
from eruption_forecast.config.forecast_config import (
    ForecastConfig,
    BaseForecastConfig,
    ForecastTrainConfig,
    ForecastPredictConfig,
    ForecastEvaluateConfig,
    ForecastCalculateConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_kwargs(output_dir: str) -> dict:
    """Return minimal valid kwargs for the new ``ForecastModel.__init__``."""
    return {
        "station": "OJN",
        "channel": "EHZ",
        "network": "VG",
        "location": "00",
        "day_to_forecast": 2,
        "output_dir": output_dir,
        "n_jobs": 4,
    }


def _full_forecast_config() -> ForecastConfig:
    """Return a fully populated ForecastConfig for serialisation tests."""
    return ForecastConfig(
        model=BaseForecastConfig(
            station="OJN",
            channel="EHZ",
            network="VG",
            location="00",
            day_to_forecast=2,
            n_jobs=4,
        ),
        calculate=ForecastCalculateConfig(
            start_date="2025-03-16",
            end_date="2025-03-22",
            source="sds",
            sds_dir="D:/Data/OJN",
            remove_outlier_method="maximum",
        ),
        train=ForecastTrainConfig(
            start_date="2025-03-16",
            end_date="2025-03-22",
            eruption_dates=["2025-03-20"],
            window_step=12,
            window_step_unit="hours",
            classifiers=["xgb"],
            cv_strategy="stratified",
            seeds=10,
        ),
        predict=ForecastPredictConfig(
            start_date="2025-03-23",
            end_date="2025-03-30",
            window_step=12,
            window_step_unit="hours",
        ),
        evaluate=ForecastEvaluateConfig(
            model="prediction",
            eruption_dates=["2025-03-20"],
        ),
    )


# ---------------------------------------------------------------------------
# BaseForecastConfig
# ---------------------------------------------------------------------------


class TestBaseForecastConfig:
    """Tests for ``BaseForecastConfig`` dataclass."""

    def test_defaults(self) -> None:
        """``BaseForecastConfig`` has expected default values."""
        cfg = BaseForecastConfig()
        assert cfg.station == ""
        assert cfg.channel == ""
        assert cfg.network == ""
        assert cfg.location == ""
        assert cfg.day_to_forecast == 2
        assert cfg.output_dir is None
        assert cfg.root_dir is None
        assert cfg.overwrite is False
        assert cfg.n_jobs == 1
        assert cfg.verbose is False

    def test_no_legacy_fields(self) -> None:
        """The new ``BaseForecastConfig`` drops legacy ``window_size`` / ``volcano_id``."""
        names = {f.name for f in fields(BaseForecastConfig)}
        assert "window_size" not in names
        assert "volcano_id" not in names
        assert "day_to_forecast" in names

    def test_to_dict_contains_all_fields(self) -> None:
        """``to_dict()`` includes every field."""
        cfg = BaseForecastConfig(station="OJN", channel="EHZ", day_to_forecast=3)
        d = cfg.to_dict()
        assert d["station"] == "OJN"
        assert d["channel"] == "EHZ"
        assert d["day_to_forecast"] == 3
        assert "n_jobs" in d

    def test_from_dict_round_trip(self) -> None:
        """``from_dict(to_dict())`` produces an equal object."""
        cfg = BaseForecastConfig(station="OJN", n_jobs=8, day_to_forecast=5)
        restored = BaseForecastConfig.from_dict(cfg.to_dict())
        assert restored.station == "OJN"
        assert restored.n_jobs == 8
        assert restored.day_to_forecast == 5

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys in the source dict do not raise."""
        cfg = BaseForecastConfig.from_dict({"station": "OJN", "unknown_key": 99})
        assert cfg.station == "OJN"

    def test_from_dict_partial(self) -> None:
        """``from_dict()`` fills unspecified fields with defaults."""
        cfg = BaseForecastConfig.from_dict({"station": "ABC"})
        assert cfg.station == "ABC"
        assert cfg.channel == ""


# ---------------------------------------------------------------------------
# ForecastCalculateConfig
# ---------------------------------------------------------------------------


class TestForecastCalculateConfig:
    """Tests for ``ForecastCalculateConfig`` dataclass."""

    def test_defaults(self) -> None:
        """``ForecastCalculateConfig`` has expected default values."""
        cfg = ForecastCalculateConfig()
        assert cfg.start_date == ""
        assert cfg.end_date == ""
        assert cfg.source == "sds"
        assert cfg.methods is None
        assert cfg.remove_outlier_method == "maximum"
        assert cfg.remove_tremor_anomalies is False
        assert cfg.interpolate is True
        assert cfg.value_multiplier is None
        assert cfg.cleanup_daily_dir is False
        assert cfg.plot_daily is False
        assert cfg.save_plot is False
        assert cfg.overwrite_plot is False
        assert cfg.sds_dir is None
        assert cfg.client_url == "https://service.iris.edu"
        assert cfg.minimum_completion_ratio == 0.3
        assert cfg.overwrite is None
        assert cfg.n_jobs is None
        assert cfg.verbose is None

    def test_to_dict_from_dict_round_trip(self) -> None:
        """``to_dict`` / ``from_dict`` round-trip preserves all values."""
        cfg = ForecastCalculateConfig(
            start_date="2025-03-16",
            end_date="2025-03-22",
            source="sds",
            sds_dir="/data/sds",
            methods=["rsam", "dsar"],
            n_jobs=8,
        )
        restored = ForecastCalculateConfig.from_dict(cfg.to_dict())
        assert restored.start_date == "2025-03-16"
        assert restored.sds_dir == "/data/sds"
        assert restored.methods == ["rsam", "dsar"]
        assert restored.n_jobs == 8

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ForecastCalculateConfig.from_dict({"source": "sds", "not_a_field": True})
        assert cfg.source == "sds"


# ---------------------------------------------------------------------------
# ForecastTrainConfig
# ---------------------------------------------------------------------------


class TestForecastTrainConfig:
    """Tests for ``ForecastTrainConfig`` dataclass."""

    def test_defaults(self) -> None:
        """``ForecastTrainConfig`` has expected default values."""
        cfg = ForecastTrainConfig()
        assert cfg.start_date == ""
        assert cfg.end_date == ""
        assert cfg.eruption_dates == []
        assert cfg.window_step == 12
        assert cfg.window_step_unit == "hours"
        assert cfg.label_builder == "standard"
        assert cfg.days_before_eruption is None
        assert cfg.classifiers == "rf"
        assert cfg.cv_strategy == "shuffle-stratified"
        assert cfg.cv_splits == 5
        assert cfg.scoring == "balanced_accuracy"
        assert cfg.top_n_features == 20
        assert cfg.include_eruption_date is True
        assert cfg.save_tremor_matrix_per_method is True
        assert cfg.minimum_completion == 1.0
        assert cfg.seeds == 10
        assert cfg.resample_method == "auto"
        assert cfg.minority_threshold == 0.15
        assert cfg.sampling_strategy == 0.75
        assert cfg.plot_features is True
        assert cfg.n_grids == 1
        assert cfg.use_cache is True

    def test_eruption_dates_is_independent(self) -> None:
        """Each instance gets its own ``eruption_dates`` list (no shared mutable default)."""
        a = ForecastTrainConfig()
        b = ForecastTrainConfig()
        a.eruption_dates.append("2025-01-01")
        assert b.eruption_dates == []

    def test_fuses_legacy_stage_params(self) -> None:
        """``ForecastTrainConfig`` carries fields from the legacy build_label/extract_features/train trio."""
        names = {f.name for f in fields(ForecastTrainConfig)}
        # From legacy BuildLabelConfig
        assert {"window_step", "window_step_unit", "eruption_dates"} <= names
        # From legacy ExtractFeaturesConfig
        assert {"select_tremor_columns", "exclude_features", "select_features"} <= names
        # From legacy ForecastTrainConfig
        assert {"classifiers", "cv_strategy", "seeds"} <= names

    def test_to_dict_from_dict_round_trip(self) -> None:
        """All fields survive a round-trip."""
        cfg = ForecastTrainConfig(
            eruption_dates=["2025-03-20", "2025-04-22"],
            classifiers=["xgb"],
            cv_strategy="stratified",
            seeds=100,
            n_jobs=4,
            select_tremor_columns=["rsam_f2", "rsam_f3"],
        )
        restored = ForecastTrainConfig.from_dict(cfg.to_dict())
        assert restored.eruption_dates == ["2025-03-20", "2025-04-22"]
        assert restored.classifiers == ["xgb"]
        assert restored.cv_strategy == "stratified"
        assert restored.seeds == 100
        assert restored.n_jobs == 4
        assert restored.select_tremor_columns == ["rsam_f2", "rsam_f3"]

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ForecastTrainConfig.from_dict({"seeds": 99, "grid_params": {"a": 1}})
        assert cfg.classifiers == "rf"
        assert cfg.seeds == 99


# ---------------------------------------------------------------------------
# ForecastPredictConfig
# ---------------------------------------------------------------------------


class TestForecastPredictConfig:
    """Tests for ``ForecastPredictConfig`` dataclass."""

    def test_defaults(self) -> None:
        """``ForecastPredictConfig`` has expected default values."""
        cfg = ForecastPredictConfig()
        assert cfg.start_date == ""
        assert cfg.end_date == ""
        assert cfg.window_step == 12
        assert cfg.window_step_unit == "hours"
        assert cfg.save_seed_result is True
        assert cfg.plot_threshold == 0.5
        assert cfg.plot_title is None
        assert cfg.plot_pdf is True
        assert cfg.use_cache is True

    def test_does_not_capture_plot_kwargs(self) -> None:
        """Variadic ``**plot_kwargs`` is deliberately excluded from the config."""
        names = {f.name for f in fields(ForecastPredictConfig)}
        assert "plot_kwargs" not in names

    def test_to_dict_from_dict_round_trip(self) -> None:
        """All fields survive a round-trip."""
        cfg = ForecastPredictConfig(
            start_date="2025-04-01",
            end_date="2025-04-07",
            window_step=6,
            plot_threshold=0.7,
            plot_pdf=False,
        )
        restored = ForecastPredictConfig.from_dict(cfg.to_dict())
        assert restored.start_date == "2025-04-01"
        assert restored.end_date == "2025-04-07"
        assert restored.window_step == 6
        assert restored.plot_threshold == 0.7
        assert restored.plot_pdf is False

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ForecastPredictConfig.from_dict({"window_step": 30, "extra": True})
        assert cfg.window_step == 30


# ---------------------------------------------------------------------------
# ForecastEvaluateConfig
# ---------------------------------------------------------------------------


class TestForecastEvaluateConfig:
    """Tests for ``ForecastEvaluateConfig`` dataclass."""

    def test_defaults(self) -> None:
        """``ForecastEvaluateConfig`` has expected default values."""
        cfg = ForecastEvaluateConfig()
        assert cfg.model == "prediction"
        assert cfg.eruption_dates is None
        assert cfg.plot_per_seed is False
        assert cfg.plot_aggregate is True

    def test_to_dict_from_dict_round_trip(self) -> None:
        """All fields survive a round-trip."""
        cfg = ForecastEvaluateConfig(
            model="training",
            eruption_dates=["2025-03-20"],
            plot_per_seed=True,
            plot_aggregate=False,
        )
        restored = ForecastEvaluateConfig.from_dict(cfg.to_dict())
        assert restored.model == "training"
        assert restored.eruption_dates == ["2025-03-20"]
        assert restored.plot_per_seed is True
        assert restored.plot_aggregate is False

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ForecastEvaluateConfig.from_dict({"model": "prediction", "extra": True})
        assert cfg.model == "prediction"


# ---------------------------------------------------------------------------
# ForecastConfig — to_dict
# ---------------------------------------------------------------------------


class TestForecastConfigToDict:
    """Tests for ``ForecastConfig.to_dict()``."""

    def test_only_model_when_no_stages_called(self) -> None:
        """``to_dict()`` omits sections that are ``None``."""
        config = ForecastConfig(model=BaseForecastConfig(station="OJN"))
        d = config.to_dict()
        assert "model" in d
        assert "version" in d
        assert "saved_at" in d
        assert "calculate" not in d
        assert "train" not in d
        assert "predict" not in d
        assert "evaluate" not in d

    def test_all_sections_present_when_set(self) -> None:
        """``to_dict()`` includes every section when all are set."""
        config = _full_forecast_config()
        d = config.to_dict()
        for key in ("model", "calculate", "train", "predict", "evaluate"):
            assert key in d

    def test_nested_values_correct(self) -> None:
        """Nested section values are serialised correctly."""
        config = _full_forecast_config()
        d = config.to_dict()
        assert d["model"]["station"] == "OJN"
        assert d["model"]["day_to_forecast"] == 2
        assert d["calculate"]["source"] == "sds"
        assert d["train"]["eruption_dates"] == ["2025-03-20"]
        assert d["train"]["classifiers"] == ["xgb"]
        assert d["predict"]["window_step"] == 12
        assert d["evaluate"]["model"] == "prediction"


# ---------------------------------------------------------------------------
# ForecastConfig — YAML save / load
# ---------------------------------------------------------------------------


class TestForecastConfigYaml:
    """Tests for ``ForecastConfig`` YAML serialisation."""

    def test_save_creates_file(self) -> None:
        """``save()`` creates the YAML file at the given path."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            returned = config.save(path)
            assert returned == path
            assert os.path.isfile(path)

    def test_yaml_contains_comment_header(self) -> None:
        """Saved YAML starts with the comment header line."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            with open(path) as f:
                first_line = f.readline().strip()
            assert first_line.startswith("# eruption-forecast")

    def test_load_full_round_trip(self) -> None:
        """``load(save())`` restores all fields exactly."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            loaded = ForecastConfig.load(path)

        assert loaded.model.station == "OJN"
        assert loaded.model.day_to_forecast == 2
        assert loaded.model.n_jobs == 4
        assert loaded.calculate is not None
        assert loaded.calculate.source == "sds"
        assert loaded.calculate.sds_dir == "D:/Data/OJN"
        assert loaded.train is not None
        assert loaded.train.eruption_dates == ["2025-03-20"]
        assert loaded.train.classifiers == ["xgb"]
        assert loaded.train.seeds == 10
        assert loaded.predict is not None
        assert loaded.predict.start_date == "2025-03-23"
        assert loaded.predict.window_step == 12
        assert loaded.evaluate is not None
        assert loaded.evaluate.model == "prediction"
        assert loaded.evaluate.eruption_dates == ["2025-03-20"]

    def test_load_partial_config(self) -> None:
        """Loading a config with only model + train sections works."""
        config = ForecastConfig(
            model=BaseForecastConfig(station="OJN"),
            train=ForecastTrainConfig(classifiers="rf", eruption_dates=["2025-03-20"]),
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "partial.yaml")
            config.save(path)
            loaded = ForecastConfig.load(path)

        assert loaded.model.station == "OJN"
        assert loaded.train is not None
        assert loaded.train.classifiers == "rf"
        assert loaded.calculate is None
        assert loaded.predict is None
        assert loaded.evaluate is None

    def test_saved_at_updated_on_save(self) -> None:
        """``saved_at`` is refreshed each time ``save()`` is called."""
        config = _full_forecast_config()
        config.saved_at = "1970-01-01T00:00:00"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            loaded = ForecastConfig.load(path)
        assert loaded.saved_at != "1970-01-01T00:00:00"

    def test_save_creates_parent_directories(self) -> None:
        """``save()`` creates nested parent directories that do not yet exist."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "deep", "config.yaml")
            config.save(path)
            assert os.path.isfile(path)

    def test_load_missing_file_raises(self) -> None:
        """``load()`` raises ``FileNotFoundError`` for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ForecastConfig.load("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# ForecastConfig — JSON save / load
# ---------------------------------------------------------------------------


class TestForecastConfigJson:
    """Tests for ``ForecastConfig`` JSON serialisation."""

    def test_save_creates_json_file(self) -> None:
        """``save(fmt='json')`` creates a valid JSON file."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"]["station"] == "OJN"

    def test_json_round_trip(self) -> None:
        """``load(save(fmt='json'))`` restores all fields."""
        config = _full_forecast_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            loaded = ForecastConfig.load(path)
        assert loaded.model.station == "OJN"
        assert loaded.train is not None
        assert loaded.train.classifiers == ["xgb"]
        assert loaded.predict is not None
        assert loaded.predict.start_date == "2025-03-23"

    def test_json_load_auto_detected_by_extension(self) -> None:
        """``load()`` auto-detects JSON format from the ``.json`` extension."""
        config = ForecastConfig(model=BaseForecastConfig(station="OJN"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            loaded = ForecastConfig.load(path)
        assert loaded.model.station == "OJN"


# ---------------------------------------------------------------------------
# ForecastModel — _config populated at __init__
# ---------------------------------------------------------------------------


class TestForecastBaseForecastConfigInit:
    """``ForecastModel`` sets ``_config.model`` at construction."""

    def test_config_model_section_matches_init_params(self) -> None:
        """``_config.model`` mirrors every ``__init__`` parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            m = fm._config.model
            assert m.station == "OJN"
            assert m.channel == "EHZ"
            assert m.network == "VG"
            assert m.location == "00"
            assert m.day_to_forecast == 2
            assert m.n_jobs == 4

    def test_config_preserves_user_output_dir(self) -> None:
        """``_config.model.output_dir`` keeps the user-supplied value, not the resolved path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            assert fm._config.model.output_dir == tmp

    def test_stage_sections_none_before_any_call(self) -> None:
        """All stage sections are ``None`` until the corresponding method runs."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            assert fm._config.calculate is None
            assert fm._config.train is None
            assert fm._config.predict is None
            assert fm._config.evaluate is None


# ---------------------------------------------------------------------------
# ForecastModel — save_config
# ---------------------------------------------------------------------------


class TestForecastModelSaveConfig:
    """Tests for ``ForecastModel.save_config()``."""

    def test_save_config_default_path_under_station_dir(self) -> None:
        """``save_config()`` without a path writes to ``{station_dir}/forecast.config.yaml``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config()
            expected = os.path.join(fm.station_dir, "forecast.config.yaml")
            assert path == expected
            assert os.path.isfile(path)

    def test_save_config_default_path_sibling_of_cache(self) -> None:
        """Default config file sits directly under ``station_dir``, next to the per-stage ``cache/`` directories."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config()
            assert os.path.basename(path) == "forecast.config.yaml"
            assert os.path.dirname(path) == fm.station_dir

    def test_save_config_custom_path(self) -> None:
        """``save_config(path)`` writes to the given path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            custom = os.path.join(tmp, "my_config.yaml")
            path = fm.save_config(custom)
            assert path == custom
            assert os.path.isfile(path)

    def test_save_config_json_format(self) -> None:
        """``save_config(fmt='json')`` produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = os.path.join(tmp, "config.json")
            fm.save_config(path, fmt="json")
            with open(path) as f:
                data = json.load(f)
            assert data["model"]["station"] == "OJN"

    def test_save_config_default_json_path(self) -> None:
        """``save_config(fmt='json')`` without ``path`` produces ``forecast.config.json``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config(fmt="json")
            assert path.endswith("forecast.config.json")
            assert os.path.isfile(path)

    def test_save_config_returns_path(self) -> None:
        """``save_config()`` returns a non-empty string path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config()
            assert isinstance(path, str)
            assert len(path) > 0

    def test_saved_yaml_readable_as_forecast_config(self) -> None:
        """YAML written by ``save_config()`` can be read back by ``ForecastConfig.load()``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            loaded = ForecastConfig.load(yaml_path)
            assert loaded.model.station == "OJN"
            assert loaded.model.day_to_forecast == 2
            assert loaded.model.n_jobs == 4


# ---------------------------------------------------------------------------
# ForecastModel — from_config
# ---------------------------------------------------------------------------


class TestForecastModelFromConfig:
    """Tests for ``ForecastModel.from_config()``."""

    def test_from_config_restores_model_params(self) -> None:
        """``from_config()`` reconstructs a ``ForecastModel`` with the saved attributes."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()

            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2.station == "OJN"
            assert fm2.channel == "EHZ"
            assert fm2.network == "VG"
            assert fm2.location == "00"
            assert fm2.day_to_forecast == 2

    def test_from_config_attaches_loaded_sections(self) -> None:
        """Stage sections present in the YAML are attached to ``_config``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            fm._config.calculate = ForecastCalculateConfig(
                start_date="2025-03-16",
                end_date="2025-03-22",
                source="sds",
                sds_dir="D:/Data/OJN",
            )
            yaml_path = fm.save_config()

            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2._config.calculate is not None
            assert fm2._config.calculate.start_date == "2025-03-16"
            assert fm2._config.calculate.sds_dir == "D:/Data/OJN"

    def test_from_config_missing_file_raises(self) -> None:
        """``from_config()`` raises ``FileNotFoundError`` for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ForecastModel.from_config("/nonexistent/config.yaml")

    def test_from_config_json_format(self) -> None:
        """``from_config()`` also works with JSON config files."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            json_path = os.path.join(tmp, "config.json")
            fm.save_config(json_path, fmt="json")
            fm2 = ForecastModel.from_config(json_path)
            assert fm2.station == "OJN"
            assert fm2.day_to_forecast == 2


# ---------------------------------------------------------------------------
# ForecastModel — run()
# ---------------------------------------------------------------------------


class TestForecastModelRun:
    """Tests for ``ForecastModel.run()``."""

    def test_run_is_noop_when_no_stages_captured(self) -> None:
        """``run()`` returns ``self`` and does nothing when every section is ``None``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            result = fm.run()
            assert result is fm
            assert fm.CalculateTremor is None
            assert fm.TrainingModel is None
            assert fm.PredictionModel is None
            assert fm.EvaluationModel is None

    def test_run_no_op_after_from_config_without_stages(self) -> None:
        """A config with no stage sections means ``run()`` is a no-op after ``from_config()``."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2.run() is fm2
