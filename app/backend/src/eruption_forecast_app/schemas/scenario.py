"""Serializable scenario configurations and sparse overrides."""

from uuid import uuid4
from typing import Any

from pydantic import Field, model_validator

from eruption_forecast_app.schemas.setup import Setup
from eruption_forecast_app.schemas.common import Parameters
from eruption_forecast_app.schemas.stages import (
    STAGES,
    Training,
    Evaluation,
    Prediction,
    Calculation,
    Explanation,
)


class Scenario(Parameters):
    id: str = Field(
        default_factory=lambda: uuid4().hex[:12], pattern=r"^[a-zA-Z0-9_-]+$"
    )
    name: str = Field(default="Scenario 1", min_length=1, max_length=120)
    description: str = ""
    training: Training = Field(default_factory=Training)
    prediction: Prediction = Field(default_factory=Prediction)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    explanation: Explanation = Field(default_factory=Explanation)


class Configuration(Parameters):
    name: str = "Research configuration"
    setup: Setup = Field(default_factory=Setup)
    calculation: Calculation = Field(default_factory=Calculation)
    training: Training = Field(default_factory=Training)
    prediction: Prediction = Field(default_factory=Prediction)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    explanation: Explanation = Field(default_factory=Explanation)
    scenarios: list[Scenario] = Field(default_factory=list)
    evaluate_model: bool = True
    explain_model: bool = True

    @model_validator(mode="after")
    def _unique_ids(self) -> "Configuration":
        if len({s.id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario IDs must be unique")
        return self


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge sparse nested overrides without mutating shared defaults.

    Example:
        >>> merge({"fit": {"seeds": 10}}, {"fit": {"seeds": 2}})
        {'fit': {'seeds': 2}}
    """
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def resolve(configuration: Configuration) -> dict[str, Any]:
    """Resolve setup defaults and sparse per-scenario overrides.

    Example:
        >>> resolve(Configuration())["setup"]["n_jobs"]
        1
    """
    result = configuration.model_dump(mode="json")
    setup = configuration.setup.model_dump(mode="json")
    for stage, model_class in STAGES.items():
        supplied = getattr(configuration, stage).model_dump(
            exclude_unset=True, mode="json"
        )
        defaults = model_class().model_dump(mode="json")
        constructor = defaults["constructor"]
        for key, value in setup.items():
            if key in constructor:
                constructor[key] = value
        if "save_model" in constructor:
            constructor["save_model"] = True
        if "nslc" in constructor:
            constructor["nslc"] = ".".join(
                setup[key] for key in ("network", "station", "location", "channel")
            )
        if "window_step" in defaults.get("label", {}):
            defaults["label"].update(window_step=12, window_step_unit="hours")
        result[stage] = merge(defaults, supplied)
        for key in (
            "n_jobs",
            "output_dir",
            "root_dir",
            "verbose",
            "network",
            "station",
            "location",
            "channel",
            "channel_type",
        ):
            if (
                key in result[stage]["constructor"]
                and result[stage]["constructor"].get(key) is None
            ):
                result[stage]["constructor"][key] = setup.get(key)
    result["scenarios"] = []
    for scenario in configuration.scenarios:
        item: dict[str, Any] = {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
        }
        for stage in ("training", "prediction", "evaluation", "explanation"):
            item[stage] = merge(
                result[stage],
                getattr(scenario, stage).model_dump(exclude_unset=True, mode="json"),
            )
            for key in ("n_jobs", "output_dir", "root_dir", "verbose"):
                if item[stage]["constructor"].get(key) is None:
                    item[stage]["constructor"][key] = setup.get(key)
        result["scenarios"].append(item)
    aliases = {
        "title": "plot_title",
        "threshold": "plot_threshold",
        "start_date": "plot_start_date",
        "end_date": "plot_end_date",
    }
    supplied_prediction = configuration.prediction.model_dump(
        exclude_unset=True, mode="json"
    )
    for source, target in aliases.items():
        if source in supplied_prediction.get(
            "plot", {}
        ) and target not in supplied_prediction.get("forecast", {}):
            result["prediction"]["forecast"][target] = supplied_prediction["plot"][
                source
            ]
    for item, scenario in zip(
        result["scenarios"], configuration.scenarios, strict=True
    ):
        supplied = scenario.prediction.model_dump(exclude_unset=True, mode="json")
        combined = merge(supplied_prediction, supplied)
        for source, target in aliases.items():
            if source in combined.get("plot", {}) and target not in combined.get(
                "forecast", {}
            ):
                item["prediction"]["forecast"][target] = combined["plot"][source]
    return result
