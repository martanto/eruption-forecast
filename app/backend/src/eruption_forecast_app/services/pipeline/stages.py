"""Scientific stage adapters, preserving package identities and persistence."""

import inspect
from typing import Any
from pathlib import Path

from matplotlib import pyplot as plt
from data_availability import SeismicAvailability

from eruption_forecast.utils.pathutils import load_pickle
from eruption_forecast.utils.feature_utils import load_select_features
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.evaluation_model import EvaluationModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor


def _identity(model: Any, config: dict, stage: str) -> dict:
    constructor = config["constructor"]
    label = dict(config["label"])
    label.pop("verbose", None)
    features = {
        key: value
        for key, value in config["features"].items()
        if key not in {"n_jobs", "verbose", "overwrite", "save_tremor_matrix_per_id"}
    }
    if stage == "prediction" and features.get("select_features") is not None:
        features["select_features"] = load_select_features(
            features["select_features"], number_of_features=0
        )
    values = {
        **constructor,
        "start_date": model.start_date,
        "end_date": model.end_date,
        "tremor_df": model.tremor_df,
        "build_label_params": label,
        "extract_features_params": features,
    }
    if stage == "training":
        values["scoring"] = config["fit"]["scoring"]
        values["fit_params"] = {
            key: config["fit"][key]
            for key in (
                "seeds",
                "resample_method",
                "minority_threshold",
                "sampling_strategy",
            )
        }
    fields = inspect.signature(type(model).build_identity).parameters
    return type(model).build_identity(**{key: values[key] for key in fields})


def _upstream(path: str) -> Any:
    value = load_pickle(path)
    if not isinstance(value, (TrainingModel, PredictionModel)):
        raise ValueError(
            "Evaluation and explanation require a saved TrainingModel or PredictionModel"
        )
    return value


def calculate(config: dict, directory: Path) -> CalculateTremor:
    """Run tremor calculation with explicit frequency bands.

    Example:
        >>> calculate(resolved['calculation'], Path('output/run'))
    """
    constructor = dict(config["constructor"])
    constructor["output_dir"] = str(directory)
    model = CalculateTremor(**constructor)
    model.change_freq_bands([tuple(band) for band in config["freq_bands"]])
    if config["source"] == "sds":
        model.from_sds(config["sds_dir"])
    else:
        model.from_fdsn(config["client_url"])
    model.run(**config["run"])
    return model


def availability(config: dict, setup: dict, directory: Path) -> None:
    """Compute real SDS completeness and save its table and calendar.

    Example:
        >>> availability(config, setup, Path('output/availability'))
    """
    constructor = config["constructor"]
    model = SeismicAvailability(
        start_date=constructor["start_date"],
        end_date=constructor["end_date"],
        sds_dir=setup["sds_dir"],
        **{
            key: setup[key]
            for key in (
                "station",
                "channel",
                "network",
                "location",
                "channel_type",
                "n_jobs",
                "verbose",
            )
        },
    )
    model.get_df().to_csv(directory / "availability.csv", index=False)
    figure = model.plot(
        title="Seismic completeness · "
        + ".".join(setup[key] for key in ("network", "station", "location", "channel"))
    )
    figure.savefig(directory / "availability.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def train(config: dict, directory: Path, tremor: Any = None) -> TrainingModel:
    """Build labels and features, then fit or restore a training cache.

    Example:
        >>> train(config, Path('output/run'), tremor_frame)
    """
    constructor = dict(config["constructor"])
    constructor["output_dir"] = str(directory)
    if tremor is not None:
        constructor["tremor_data"] = tremor
    model = TrainingModel(**constructor)
    external = config["load_features"].get("features_matrix_path")
    if config["use_cache"] and not constructor["overwrite"] and not external:
        cached = TrainingModel.load(
            str(directory / "training"), _identity(model, config, "training")
        )
        if cached is not None:
            return cached
    if external:
        model.load_features(**config["load_features"])
    else:
        model.build_label(**config["label"])
        model.extract_features(**config["features"])
    model.fit(**config["fit"])
    return model


def predict(
    config: dict,
    directory: Path,
    training: TrainingModel | None = None,
    tremor: Any = None,
) -> PredictionModel:
    """Forecast with explicit feature selection and upstream cache identity.

    Example:
        >>> predict(config, Path('output/run'), training, tremor_frame)
    """
    constructor = dict(config["constructor"])
    constructor["output_dir"] = str(directory)
    if training is None and constructor.get("model", "").endswith(".pkl"):
        candidate = load_pickle(constructor["model"])
        if isinstance(candidate, TrainingModel):
            training = candidate
    if training is not None:
        constructor.update(model=training.ClassifierEnsemble, training_hash=training.id)
    if tremor is not None:
        constructor["tremor_data"] = tremor
    model = PredictionModel(**constructor)
    features = dict(config["features"])
    if config["use_features_from"] == "training":
        if training is None or training.ClassifierEnsemble is None:
            raise ValueError(
                "Training feature mode requires a complete saved TrainingModel, not a bare ensemble"
            )
        features["select_features"] = training.ClassifierEnsemble.features
    effective = {
        **config,
        "constructor": {**constructor, "model": None},
        "features": features,
    }
    if (
        config["use_cache"]
        and not constructor["overwrite"]
        and config["use_features_from"] != "files"
    ):
        cached = PredictionModel.load(
            str(directory / "prediction"), _identity(model, effective, "prediction")
        )
        if cached is not None:
            return cached
    if config["use_features_from"] == "files":
        kwargs = dict(config["load_features"])
        kwargs["window_step"] = kwargs["window_step"] or config["label"]["window_step"]
        kwargs["window_step_unit"] = (
            kwargs["window_step_unit"] or config["label"]["window_step_unit"]
        )
        model.load_features(**kwargs)
    else:
        model.build_label(**config["label"])
        model.extract_features(**features)
    plot = {
        key: value
        for key, value in config["plot"].items()
        if key not in {"title", "threshold", "start_date", "end_date"}
    }
    if config["enable_segments_plot"] and training:
        plot.update(
            training_start_date=training.start_date_str,
            training_end_date=training.end_date_str,
            prediction_start_date=model.start_date_str,
            prediction_end_date=model.end_date_str,
        )
    model.forecast(**config["forecast"], **plot)
    return model


def evaluate(config: dict, directory: Path, upstream: Any = None) -> EvaluationModel:
    """Evaluate a fitted model without retraining.

    Example:
        >>> evaluate(config, Path('output/run'), prediction)
    """
    constructor = dict(config["constructor"])
    constructor["output_dir"] = str(directory)
    constructor["model"] = (
        upstream if upstream is not None else _upstream(constructor["model"])
    )
    model = EvaluationModel(**constructor)
    if model.model_kind == "prediction":
        kwargs = dict(config["label"])
        kwargs["window_step"] = kwargs["window_step"] or model.model.window_step
        kwargs["window_step_unit"] = (
            kwargs["window_step_unit"] or model.model.window_step_unit
        )
        model.build_label(**kwargs)
    model.evaluate(**config["execute"])
    if config["execute"]["compare_classifiers"]:
        model.compare(**config["compare"])
    return model


def explain(config: dict, directory: Path, upstream: Any = None) -> ExplanationModel:
    """Compute package SHAP payloads and requested figures.

    Example:
        >>> explain(config, Path('output/run'), prediction)
    """
    constructor = dict(config["constructor"])
    constructor["output_dir"] = str(directory)
    constructor["model"] = (
        upstream if upstream is not None else _upstream(constructor["model"])
    )
    model = ExplanationModel(**constructor)
    model.explain(**config["execute"])
    model.plot(**config["plot"])
    return model
