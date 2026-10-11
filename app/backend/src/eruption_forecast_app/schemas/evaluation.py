"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from typing import Literal

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class EvaluationConstructor(Parameters):
    """Parameters for ``model/evaluation_model.py:__init__``."""

    model: str | None = Field(
        default=None, description="Model. Package option: __init__."
    )
    eruption_dates: list[str] | None = Field(
        default=None, description="Eruption dates. Package option: __init__."
    )
    overwrite: bool | None = Field(
        default=False, description="Overwrite. Package option: __init__."
    )
    output_dir: str | None = Field(
        default=None, description="Output dir. Package option: __init__."
    )
    root_dir: str | None = Field(
        default=None, description="Root dir. Package option: __init__."
    )
    prefix_config: str | None = Field(
        default=None, description="Prefix config. Package option: __init__."
    )
    n_jobs: int | None = Field(
        default=1, description="N jobs. Package option: __init__."
    )
    save_model: bool = Field(
        default=False, description="Save model. Package option: __init__."
    )
    verbose: bool | None = Field(
        default=False, description="Verbose. Package option: __init__."
    )


class EvaluationLabel(Parameters):
    """Parameters for ``model/evaluation_model.py:build_label``."""

    window_step: int | None = Field(
        default=None, description="Window step. Package option: build_label."
    )
    window_step_unit: Literal["minutes", "hours"] | None = Field(
        default=None, description="Window step unit. Package option: build_label."
    )


class EvaluationExecute(Parameters):
    """Parameters for ``model/evaluation_model.py:evaluate``."""

    plot_aggregate: bool = Field(
        default=True, description="Plot aggregate. Package option: evaluate."
    )
    plot_per_seed: bool = Field(
        default=False, description="Plot per seed. Package option: evaluate."
    )
    compare_classifiers: bool = Field(
        default=True, description="Compare classifiers. Package option: evaluate."
    )
    use_cache: bool = Field(
        default=True, description="Use cache. Package option: evaluate."
    )


class EvaluationCompare(Parameters):
    """Parameters for ``model/evaluation_model.py:compare``."""

    metrics: str | list[str] | None = Field(
        default=None, description="Metrics. Package option: compare."
    )
