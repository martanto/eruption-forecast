"""Tests for the per-stage config dataclasses.

Two concerns are covered:

1. **Drift guard** — each config must mirror its owner's ``__init__``
   surface (CLAUDE.md Rule 16 for ``TrainingConfig``; the same contract holds
   for the other stage configs and ``LabelConfig``). A new, renamed, or
   re-defaulted constructor parameter fails here until the config catches up.
2. **Persistence** — ``save()`` / ``load()`` round-trip through YAML and JSON,
   ``to_dict()`` leads with the version stamp, and ``from_dict()`` tolerates
   unknown keys.
"""

import json
import inspect
import dataclasses
from typing import Any

import yaml
import pytest

from eruption_forecast.config.base_config import BaseConfig
from eruption_forecast.config.label_config import LabelConfig
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.config.training_config import TrainingConfig
from eruption_forecast.model.evaluation_model import EvaluationModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.config.evaluation_config import EvaluationConfig
from eruption_forecast.config.prediction_config import PredictionConfig
from eruption_forecast.config.explanation_config import ExplanationConfig


METADATA_FIELDS = {"version", "saved_at"}

# Constructor parameters that are intentionally not captured because they
# hold live objects (fitted models) that cannot round-trip through YAML.
NON_SERIALISABLE_PARAMS: dict[type, set[str]] = {
    EvaluationModel: {"model"},
    ExplanationModel: {"model"},
}

CONFIG_PAIRS: list[tuple[type, type[BaseConfig]]] = [
    (TrainingModel, TrainingConfig),
    (PredictionModel, PredictionConfig),
    (EvaluationModel, EvaluationConfig),
    (ExplanationModel, ExplanationConfig),
    (LabelBuilder, LabelConfig),
]

CONFIG_IDS = [config.__name__ for _, config in CONFIG_PAIRS]
CONFIG_CLASSES = [config for _, config in CONFIG_PAIRS]


def _init_defaults(owner: type) -> dict[str, Any]:
    """Return ``{name: default}`` for every named ``__init__`` parameter."""
    skip = NON_SERIALISABLE_PARAMS.get(owner, set())
    return {
        name: param.default
        for name, param in inspect.signature(owner.__init__).parameters.items()
        if name != "self"
        and name not in skip
        and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }


def _config_defaults(config: type[BaseConfig]) -> dict[str, Any]:
    """Return ``{name: default}`` for every non-metadata config field."""
    defaults: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        if f.name in METADATA_FIELDS:
            continue
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
        else:
            defaults[f.name] = dataclasses.MISSING
    return defaults


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("owner", "config"), CONFIG_PAIRS, ids=CONFIG_IDS)
def test_config_fields_match_init_parameters(owner: type, config: type) -> None:
    init_names = set(_init_defaults(owner))
    config_names = set(_config_defaults(config))
    assert init_names - config_names == set(), "Missing from config"
    assert config_names - init_names == set(), "Not accepted by __init__"


@pytest.mark.parametrize(("owner", "config"), CONFIG_PAIRS, ids=CONFIG_IDS)
def test_config_defaults_match_init_defaults(owner: type, config: type) -> None:
    init_defaults = _init_defaults(owner)
    config_defaults = _config_defaults(config)
    for name, init_default in init_defaults.items():
        config_default = config_defaults[name]
        if init_default is inspect.Parameter.empty:
            # Required constructor args get a neutral placeholder in the config.
            continue
        if init_default is None and config_default in ([], ""):
            # ``None`` list/date args are stored as empty containers so the
            # YAML stays typed.
            continue
        assert config_default == init_default, (
            f"{config.__name__}.{name} default {config_default!r} != "
            f"{owner.__name__}.__init__ default {init_default!r}"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_to_dict_leads_with_metadata(config: type[BaseConfig]) -> None:
    data = config().to_dict()
    assert list(data)[:2] == ["version", "saved_at"]
    assert set(_config_defaults(config)) <= set(data)


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_to_init_kwargs_strips_metadata(config: type[BaseConfig]) -> None:
    kwargs = config().to_init_kwargs()
    assert METADATA_FIELDS.isdisjoint(kwargs)


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_from_dict_ignores_unknown_keys(config: type[BaseConfig]) -> None:
    instance = config.from_dict({"unknown_key": 1, "version": "9.9.9"})
    assert instance.version == "9.9.9"
    assert not hasattr(instance, "unknown_key")


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_save_load_round_trip(config: type[BaseConfig], fmt: str, tmp_path) -> None:
    # Flip every bool field so the round trip proves non-default values survive.
    flipped = {
        name: not value
        for name, value in _config_defaults(config).items()
        if isinstance(value, bool)
    }
    assert flipped, f"{config.__name__} has no bool field to flip"
    original = config(**flipped)
    path = tmp_path / "nested" / f"stage.config.{fmt}"

    written = original.save(str(path), fmt=fmt)  # type: ignore[attr-defined]
    loaded = config.load(written)  # type: ignore[attr-defined]

    assert path.is_file()
    assert loaded.to_dict() == original.to_dict()


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_yaml_has_header_comment(config: type[BaseConfig], tmp_path) -> None:
    path = config().save(str(tmp_path / "c.yaml"))  # type: ignore[attr-defined]
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
        body = yaml.safe_load(f)
    assert first_line.startswith("# eruption-forecast")
    assert body["version"]


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_json_is_valid(config: type[BaseConfig], tmp_path) -> None:
    path = config().save(str(tmp_path / "c.json"), fmt="json")  # type: ignore[attr-defined]
    with open(path, encoding="utf-8") as f:
        assert "saved_at" in json.load(f)


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_load_missing_file_raises(config: type[BaseConfig], tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        config.load(str(tmp_path / "missing.yaml"))  # type: ignore[attr-defined]


@pytest.mark.parametrize("config", CONFIG_CLASSES, ids=CONFIG_IDS)
def test_load_empty_yaml_returns_defaults(config: type[BaseConfig], tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    loaded = config.load(str(path))  # type: ignore[attr-defined]
    assert loaded.to_init_kwargs() == config().to_init_kwargs()


def test_save_refreshes_saved_at(tmp_path) -> None:
    config = TrainingConfig(saved_at="2000-01-01T00:00:00")
    config.save(str(tmp_path / "t.yaml"))
    assert config.saved_at != "2000-01-01T00:00:00"
