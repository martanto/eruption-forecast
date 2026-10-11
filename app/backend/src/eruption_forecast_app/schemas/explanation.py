"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class ExplanationConstructor(Parameters):
    """Parameters for ``model/explanation_model.py:__init__``."""

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


class ExplanationExecute(Parameters):
    """Parameters for ``model/explanation_model.py:explain``."""

    save_per_seed: bool = Field(
        default=True, description="Save per seed. Package option: explain."
    )
    check_additivity: bool = Field(
        default=False, description="Check additivity. Package option: explain."
    )
    overwrite_classifier_explanation: bool = Field(
        default=False,
        description="Overwrite classifier explanation. Package option: explain.",
    )
    use_cache: bool = Field(
        default=True, description="Use cache. Package option: explain."
    )


class ExplanationPlot(Parameters):
    """Parameters for ``model/explanation_model.py:plot``."""

    figsize: tuple[float, float] | None = Field(
        default=None, description="Figsize. Package option: plot."
    )
    max_display: int = Field(
        default=20, description="Max display. Package option: plot."
    )
    group_remaining_features: bool = Field(
        default=False, description="Group remaining features. Package option: plot."
    )
    dpi: int = Field(default=150, description="Dpi. Package option: plot.")
    plot_per_seed: bool = Field(
        default=True, description="Plot per seed. Package option: plot."
    )
    plot_aggregate: bool = Field(
        default=True, description="Plot aggregate. Package option: plot."
    )
