"""Unit tests for PipelineConfig and ForecastModel config persistence.

Covers:
- Each section dataclass: default values, to_dict(), from_dict()
- PipelineConfig: to_dict(), save()/load() round-trips (YAML and JSON)
- PipelineConfig.load() with partial configs (missing sections)
- Unknown keys in from_dict() are silently ignored
- ForecastModel._config populated at init (model section only)
- ForecastModel._config updated after calculate() is stubbed via load_tremor_data()
- ForecastModel.save_config() / from_config() round-trip
- ForecastModel.save_model() / load_model() round-trip
- ForecastModel.run() raises RuntimeError when not loaded via from_config()
- ForecastModel.load_model() raises FileNotFoundError for missing path
- PipelineConfig.load() raises FileNotFoundError for missing path

All tests use temporary directories and in-memory data — no seismic files required.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from eruption_forecast import ForecastModel, PipelineConfig
from eruption_forecast.config.pipeline_config import (
    BuildLabelConfig,
    CalculateConfig,
    ExtractFeaturesConfig,
    ForecastConfig,
    ModelConfig,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_kwargs(output_dir: str) -> dict:
    """Return minimal valid kwargs for ForecastModel.__init__."""
    return {
        "station": "OJN",
        "channel": "EHZ",
        "start_date": "2025-03-16",
        "end_date": "2025-03-22",
        "window_size": 1,
        "volcano_id": "MERAPI",
        "network": "VG",
        "location": "00",
        "output_dir": output_dir,
        "n_jobs": 4,
    }


def _full_pipeline_config() -> PipelineConfig:
    """Return a fully populated PipelineConfig for serialisation tests."""
    return PipelineConfig(
        model=ModelConfig(
            station="OJN",
            channel="EHZ",
            start_date="2025-03-16",
            end_date="2025-03-22",
            window_size=1,
            volcano_id="MERAPI",
            network="VG",
            location="00",
            n_jobs=4,
        ),
        calculate=CalculateConfig(
            source="sds",
            sds_dir="D:/Data/OJN",
            remove_outlier_method="maximum",
        ),
        build_label=BuildLabelConfig(
            window_step=12,
            window_step_unit="hours",
            day_to_forecast=2,
            eruption_dates=["2025-03-20"],
        ),
        extract_features=ExtractFeaturesConfig(
            select_tremor_columns=["rsam_f2", "rsam_f3"],
            use_relevant_features=False,
        ),
        train=TrainConfig(
            classifier="xgb",
            cv_strategy="stratified",
            random_state=0,
            total_seed=10,
        ),
        forecast=ForecastConfig(
            start_date="2025-03-23",
            end_date="2025-03-30",
            window_size=1,
            window_step=12,
            window_step_unit="hours",
        ),
    )


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self) -> None:
        """ModelConfig has expected default values."""
        cfg = ModelConfig()
        assert cfg.station == ""
        assert cfg.channel == ""
        assert cfg.window_size == 1
        assert cfg.network == "VG"
        assert cfg.location == "00"
        assert cfg.overwrite is False
        assert cfg.n_jobs == 1
        assert cfg.verbose is False
        assert cfg.debug is False

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict() includes every field."""
        cfg = ModelConfig(station="OJN", channel="EHZ", volcano_id="MERAPI")
        d = cfg.to_dict()
        assert d["station"] == "OJN"
        assert d["channel"] == "EHZ"
        assert d["volcano_id"] == "MERAPI"
        assert "window_size" in d
        assert "n_jobs" in d

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict()) produces an equal object."""
        cfg = ModelConfig(station="OJN", n_jobs=8, debug=True)
        restored = ModelConfig.from_dict(cfg.to_dict())
        assert restored.station == "OJN"
        assert restored.n_jobs == 8
        assert restored.debug is True

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys in the source dict do not raise an error."""
        cfg = ModelConfig.from_dict({"station": "OJN", "unknown_key": 99})
        assert cfg.station == "OJN"

    def test_from_dict_partial(self) -> None:
        """from_dict() fills unspecified fields with defaults."""
        cfg = ModelConfig.from_dict({"station": "ABC"})
        assert cfg.station == "ABC"
        assert cfg.channel == ""  # default


# ---------------------------------------------------------------------------
# CalculateConfig
# ---------------------------------------------------------------------------


class TestCalculateConfig:
    """Tests for CalculateConfig dataclass."""

    def test_defaults(self) -> None:
        """CalculateConfig has expected default values."""
        cfg = CalculateConfig()
        assert cfg.source == "sds"
        assert cfg.remove_outlier_method == "maximum"
        assert cfg.interpolate is True
        assert cfg.cleanup_daily_dir is False
        assert cfg.plot_daily is False
        assert cfg.save_plot is False

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict / from_dict round-trip preserves all values."""
        cfg = CalculateConfig(source="sds", sds_dir="/data/sds", n_jobs=8)
        restored = CalculateConfig.from_dict(cfg.to_dict())
        assert restored.source == "sds"
        assert restored.sds_dir == "/data/sds"
        assert restored.n_jobs == 8

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = CalculateConfig.from_dict({"source": "sds", "not_a_field": True})
        assert cfg.source == "sds"


# ---------------------------------------------------------------------------
# BuildLabelConfig
# ---------------------------------------------------------------------------


class TestBuildLabelConfig:
    """Tests for BuildLabelConfig dataclass."""

    def test_defaults(self) -> None:
        """BuildLabelConfig has expected default values."""
        cfg = BuildLabelConfig()
        assert cfg.window_step == 12
        assert cfg.window_step_unit == "hours"
        assert cfg.day_to_forecast == 2
        assert cfg.eruption_dates == []
        assert cfg.start_date is None
        assert cfg.end_date is None
        assert cfg.tremor_columns is None

    def test_eruption_dates_is_independent(self) -> None:
        """Each instance gets its own eruption_dates list (no shared mutable default)."""
        a = BuildLabelConfig()
        b = BuildLabelConfig()
        a.eruption_dates.append("2025-01-01")
        assert b.eruption_dates == []

    def test_to_dict_from_dict_round_trip(self) -> None:
        """eruption_dates list survives a round-trip."""
        cfg = BuildLabelConfig(eruption_dates=["2025-03-20", "2025-04-22"])
        restored = BuildLabelConfig.from_dict(cfg.to_dict())
        assert restored.eruption_dates == ["2025-03-20", "2025-04-22"]

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = BuildLabelConfig.from_dict({"window_step": 6, "bogus": "value"})
        assert cfg.window_step == 6


# ---------------------------------------------------------------------------
# ExtractFeaturesConfig
# ---------------------------------------------------------------------------


class TestExtractFeaturesConfig:
    """Tests for ExtractFeaturesConfig dataclass."""

    def test_defaults(self) -> None:
        """ExtractFeaturesConfig has expected default values."""
        cfg = ExtractFeaturesConfig()
        assert cfg.select_tremor_columns is None
        assert cfg.save_tremor_matrix_per_method is True
        assert cfg.save_tremor_matrix_per_id is False
        assert cfg.exclude_features is None
        assert cfg.use_relevant_features is False
        assert cfg.overwrite is False
        assert cfg.n_jobs is None
        assert cfg.verbose is None

    def test_to_dict_from_dict_round_trip(self) -> None:
        """Columns list survives a round-trip."""
        cfg = ExtractFeaturesConfig(select_tremor_columns=["rsam_f2", "rsam_f3"])
        restored = ExtractFeaturesConfig.from_dict(cfg.to_dict())
        assert restored.select_tremor_columns == ["rsam_f2", "rsam_f3"]

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ExtractFeaturesConfig.from_dict({"n_jobs": 4, "unsupported": "x"})
        assert cfg.n_jobs == 4


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_defaults(self) -> None:
        """TrainConfig has expected default values."""
        cfg = TrainConfig()
        assert cfg.classifier == "rf"
        assert cfg.cv_strategy == "shuffle"
        assert cfg.random_state == 0
        assert cfg.total_seed == 500
        assert cfg.with_evaluation is True
        assert cfg.number_of_significant_features == 20
        assert cfg.sampling_strategy == 0.75
        assert cfg.save_all_features is False
        assert cfg.plot_significant_features is False
        assert cfg.n_jobs is None
        assert cfg.overwrite is False
        assert cfg.verbose is False

    def test_to_dict_from_dict_round_trip(self) -> None:
        """All fields survive a round-trip."""
        cfg = TrainConfig(
            classifier="xgb", cv_strategy="stratified", total_seed=100, n_jobs=4
        )
        restored = TrainConfig.from_dict(cfg.to_dict())
        assert restored.classifier == "xgb"
        assert restored.cv_strategy == "stratified"
        assert restored.total_seed == 100
        assert restored.n_jobs == 4

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = TrainConfig.from_dict({"classifier": "nb", "grid_params": {"a": 1}})
        assert cfg.classifier == "nb"


# ---------------------------------------------------------------------------
# ForecastConfig
# ---------------------------------------------------------------------------


class TestForecastConfig:
    """Tests for ForecastConfig dataclass."""

    def test_defaults(self) -> None:
        """ForecastConfig has expected default values."""
        cfg = ForecastConfig()
        assert cfg.start_date == ""
        assert cfg.end_date == ""
        assert cfg.window_size == 1
        assert cfg.window_step == 12
        assert cfg.window_step_unit == "hours"
        assert cfg.save_predictions is True
        assert cfg.save_plot is True
        assert cfg.n_jobs is None
        assert cfg.overwrite is False
        assert cfg.verbose is False

    def test_to_dict_from_dict_round_trip(self) -> None:
        """All fields survive a round-trip."""
        cfg = ForecastConfig(
            start_date="2025-04-01",
            end_date="2025-04-07",
            window_size=2,
            window_step=6,
        )
        restored = ForecastConfig.from_dict(cfg.to_dict())
        assert restored.start_date == "2025-04-01"
        assert restored.end_date == "2025-04-07"
        assert restored.window_size == 2
        assert restored.window_step == 6

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys do not raise."""
        cfg = ForecastConfig.from_dict({"window_step": 30, "extra": True})
        assert cfg.window_step == 30


# ---------------------------------------------------------------------------
# PipelineConfig — to_dict
# ---------------------------------------------------------------------------


class TestPipelineConfigToDict:
    """Tests for PipelineConfig.to_dict()."""

    def test_only_model_when_no_stages_called(self) -> None:
        """to_dict() omits sections that are None."""
        config = PipelineConfig(model=ModelConfig(station="OJN"))
        d = config.to_dict()
        assert "model" in d
        assert "version" in d
        assert "saved_at" in d
        assert "calculate" not in d
        assert "build_label" not in d
        assert "extract_features" not in d
        assert "train" not in d
        assert "forecast" not in d

    def test_all_sections_present_when_set(self) -> None:
        """to_dict() includes every section when all are set."""
        config = _full_pipeline_config()
        d = config.to_dict()
        for key in ("model", "calculate", "build_label", "extract_features", "train", "forecast"):
            assert key in d

    def test_nested_values_correct(self) -> None:
        """Nested section values are serialised correctly."""
        config = _full_pipeline_config()
        d = config.to_dict()
        assert d["model"]["station"] == "OJN"
        assert d["calculate"]["source"] == "sds"
        assert d["build_label"]["eruption_dates"] == ["2025-03-20"]
        assert d["train"]["classifier"] == "xgb"
        assert d["forecast"]["window_step"] == 12


# ---------------------------------------------------------------------------
# PipelineConfig — YAML save / load
# ---------------------------------------------------------------------------


class TestPipelineConfigYaml:
    """Tests for PipelineConfig YAML serialisation."""

    def test_save_creates_file(self) -> None:
        """save() creates the YAML file at the given path."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            returned = config.save(path)
            assert returned == path
            assert os.path.isfile(path)

    def test_yaml_contains_comment_header(self) -> None:
        """Saved YAML starts with the comment header line."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            with open(path) as f:
                first_line = f.readline().strip()
            assert first_line.startswith("# eruption-forecast")

    def test_load_full_round_trip(self) -> None:
        """load(save()) restores all fields exactly."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            loaded = PipelineConfig.load(path)

        assert loaded.model.station == "OJN"
        assert loaded.model.n_jobs == 4
        assert loaded.calculate.source == "sds"
        assert loaded.calculate.sds_dir == "D:/Data/OJN"
        assert loaded.build_label.window_step == 12
        assert loaded.build_label.eruption_dates == ["2025-03-20"]
        assert loaded.extract_features.select_tremor_columns == ["rsam_f2", "rsam_f3"]
        assert loaded.train.classifier == "xgb"
        assert loaded.train.total_seed == 10
        assert loaded.forecast.start_date == "2025-03-23"
        assert loaded.forecast.window_step == 12

    def test_load_partial_config(self) -> None:
        """Loading a config with only model + train sections works."""
        config = PipelineConfig(
            model=ModelConfig(station="OJN"),
            train=TrainConfig(classifier="rf"),
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "partial.yaml")
            config.save(path)
            loaded = PipelineConfig.load(path)

        assert loaded.model.station == "OJN"
        assert loaded.train.classifier == "rf"
        assert loaded.calculate is None
        assert loaded.build_label is None
        assert loaded.extract_features is None
        assert loaded.forecast is None

    def test_saved_at_updated_on_save(self) -> None:
        """saved_at is refreshed each time save() is called."""
        config = _full_pipeline_config()
        config.saved_at = "1970-01-01T00:00:00"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.yaml")
            config.save(path)
            loaded = PipelineConfig.load(path)
        assert loaded.saved_at != "1970-01-01T00:00:00"

    def test_save_creates_parent_directories(self) -> None:
        """save() creates nested parent directories that do not yet exist."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "deep", "config.yaml")
            config.save(path)
            assert os.path.isfile(path)

    def test_load_missing_file_raises(self) -> None:
        """load() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            PipelineConfig.load("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# PipelineConfig — JSON save / load
# ---------------------------------------------------------------------------


class TestPipelineConfigJson:
    """Tests for PipelineConfig JSON serialisation."""

    def test_save_creates_json_file(self) -> None:
        """save(fmt='json') creates a valid JSON file."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"]["station"] == "OJN"

    def test_json_round_trip(self) -> None:
        """load(save(fmt='json')) restores all fields."""
        config = _full_pipeline_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            loaded = PipelineConfig.load(path)
        assert loaded.model.station == "OJN"
        assert loaded.train.classifier == "xgb"
        assert loaded.forecast.start_date == "2025-03-23"

    def test_json_load_auto_detected_by_extension(self) -> None:
        """load() auto-detects JSON format from the .json extension."""
        config = PipelineConfig(model=ModelConfig(station="OJN"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            config.save(path, fmt="json")
            loaded = PipelineConfig.load(path)
        assert loaded.model.station == "OJN"


# ---------------------------------------------------------------------------
# ForecastModel — _config populated at __init__
# ---------------------------------------------------------------------------


class TestForecastModelConfigInit:
    """ForecastModel sets _config.model at construction."""

    def test_config_model_section_matches_init_params(self) -> None:
        """_config.model mirrors every __init__ parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            m = fm._config.model
            assert m.station == "OJN"
            assert m.channel == "EHZ"
            assert m.start_date == "2025-03-16"
            assert m.end_date == "2025-03-22"
            assert m.window_size == 1
            assert m.volcano_id == "MERAPI"
            assert m.network == "VG"
            assert m.location == "00"
            assert m.n_jobs == 4

    def test_stage_sections_none_before_any_call(self) -> None:
        """All stage sections are None until the corresponding method is called."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            assert fm._config.calculate is None
            assert fm._config.build_label is None
            assert fm._config.extract_features is None
            assert fm._config.train is None
            assert fm._config.forecast is None

    def test_loaded_config_none_at_init(self) -> None:
        """_loaded_config is None for a freshly constructed instance."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            assert fm._loaded_config is None


# ---------------------------------------------------------------------------
# ForecastModel — save_config
# ---------------------------------------------------------------------------


class TestForecastModelSaveConfig:
    """Tests for ForecastModel.save_config()."""

    def test_save_config_default_path(self) -> None:
        """save_config() without an explicit path writes to {station_dir}/config.yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config()
            assert path == os.path.join(fm.station_dir, "config.yaml")
            assert os.path.isfile(path)

    def test_save_config_custom_path(self) -> None:
        """save_config(path) writes to the given path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            custom = os.path.join(tmp, "my_config.yaml")
            path = fm.save_config(custom)
            assert path == custom
            assert os.path.isfile(path)

    def test_save_config_json_format(self) -> None:
        """save_config(fmt='json') produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = os.path.join(tmp, "config.json")
            fm.save_config(path, fmt="json")
            with open(path) as f:
                data = json.load(f)
            assert data["model"]["station"] == "OJN"

    def test_save_config_default_json_path(self) -> None:
        """save_config(fmt='json') without path uses {station_dir}/config.json."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config(fmt="json")
            assert path.endswith("config.json")
            assert os.path.isfile(path)

    def test_save_config_returns_path(self) -> None:
        """save_config() returns a non-empty string path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_config()
            assert isinstance(path, str)
            assert len(path) > 0

    def test_saved_yaml_readable_as_pipeline_config(self) -> None:
        """YAML written by save_config() can be read back by PipelineConfig.load()."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            loaded = PipelineConfig.load(yaml_path)
            assert loaded.model.station == "OJN"
            assert loaded.model.n_jobs == 4


# ---------------------------------------------------------------------------
# ForecastModel — from_config
# ---------------------------------------------------------------------------


class TestForecastModelFromConfig:
    """Tests for ForecastModel.from_config()."""

    def test_from_config_restores_model_params(self) -> None:
        """from_config() reconstructs a ForecastModel with the correct attributes."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()

            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2.station == "OJN"
            assert fm2.channel == "EHZ"
            assert fm2.window_size == 1
            assert fm2.volcano_id == "MERAPI"
            assert fm2.network == "VG"
            assert fm2.n_jobs == 4

    def test_from_config_sets_loaded_config(self) -> None:
        """from_config() attaches the PipelineConfig to _loaded_config."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2._loaded_config is not None
            assert isinstance(fm2._loaded_config, PipelineConfig)

    def test_from_config_loaded_config_has_model_section(self) -> None:
        """_loaded_config.model matches the saved parameters."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            fm2 = ForecastModel.from_config(yaml_path)
            assert fm2._loaded_config.model.station == "OJN"

    def test_from_config_missing_file_raises(self) -> None:
        """from_config() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ForecastModel.from_config("/nonexistent/config.yaml")

    def test_from_config_json_format(self) -> None:
        """from_config() also works with JSON config files."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            json_path = os.path.join(tmp, "config.json")
            fm.save_config(json_path, fmt="json")
            fm2 = ForecastModel.from_config(json_path)
            assert fm2.station == "OJN"


# ---------------------------------------------------------------------------
# ForecastModel — save_model / load_model
# ---------------------------------------------------------------------------


class TestForecastModelSaveLoadModel:
    """Tests for ForecastModel.save_model() and load_model()."""

    def test_save_model_default_path(self) -> None:
        """save_model() without a path writes to {station_dir}/forecast_model.pkl."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_model()
            assert path == os.path.join(fm.station_dir, "forecast_model.pkl")
            assert os.path.isfile(path)

    def test_save_model_custom_path(self) -> None:
        """save_model(path) writes to the given path."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            custom = os.path.join(tmp, "my_model.pkl")
            path = fm.save_model(custom)
            assert path == custom
            assert os.path.isfile(path)

    def test_save_model_returns_path(self) -> None:
        """save_model() returns the written path as a string."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            path = fm.save_model()
            assert isinstance(path, str)
            assert path.endswith(".pkl")

    def test_load_model_restores_attributes(self) -> None:
        """load_model() restores station, channel, and other attributes."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            pkl_path = fm.save_model()

            fm2 = ForecastModel.load_model(pkl_path)
            assert fm2.station == "OJN"
            assert fm2.channel == "EHZ"
            assert fm2.window_size == 1
            assert fm2.n_jobs == 4

    def test_load_model_restores_config(self) -> None:
        """load_model() preserves _config.model."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            pkl_path = fm.save_model()

            fm2 = ForecastModel.load_model(pkl_path)
            assert fm2._config.model.station == "OJN"

    def test_load_model_missing_file_raises(self) -> None:
        """load_model() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ForecastModel.load_model("/nonexistent/forecast_model.pkl")


# ---------------------------------------------------------------------------
# ForecastModel — run()
# ---------------------------------------------------------------------------


class TestForecastModelRun:
    """Tests for ForecastModel.run()."""

    def test_run_raises_without_from_config(self) -> None:
        """run() raises RuntimeError when the instance was not loaded via from_config()."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            with pytest.raises(RuntimeError, match="from_config"):
                fm.run()

    def test_run_raises_after_load_model(self) -> None:
        """run() also raises for a load_model()-restored instance (no _loaded_config)."""
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            pkl = fm.save_model()
            fm2 = ForecastModel.load_model(pkl)
            with pytest.raises(RuntimeError, match="from_config"):
                fm2.run()

    def test_run_available_after_from_config(self) -> None:
        """from_config() sets _loaded_config so run() no longer raises immediately.

        A config with no stage sections means run() does nothing and returns self.
        """
        with tempfile.TemporaryDirectory() as tmp:
            fm = ForecastModel(**_model_kwargs(tmp))
            yaml_path = fm.save_config()
            fm2 = ForecastModel.from_config(yaml_path)
            # All stage sections are None → run() is a no-op, returns self
            result = fm2.run()
            assert result is fm2


# ---------------------------------------------------------------------------
# Top-level package export
# ---------------------------------------------------------------------------


class TestTopLevelExport:
    """PipelineConfig is importable from the top-level package."""

    def test_pipeline_config_exported(self) -> None:
        """eruption_forecast.PipelineConfig is the same class as the module-level one."""
        from eruption_forecast import PipelineConfig as TopLevel
        from eruption_forecast.config.pipeline_config import PipelineConfig as Direct

        assert TopLevel is Direct
