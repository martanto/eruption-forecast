"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from typing import Literal

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class PredictionConstructor(Parameters):
    """Parameters for ``model/prediction_model.py:__init__``."""

    model: str | None = Field(
        default=None, description="Model. Package option: __init__."
    )
    tremor_data: str | None = Field(
        default=None, description="Tremor data. Package option: __init__."
    )
    start_date: str | None = Field(
        default=None, description="Start date. Package option: __init__."
    )
    end_date: str | None = Field(
        default=None, description="End date. Package option: __init__."
    )
    window_size: int = Field(
        default=2, description="Window size. Package option: __init__."
    )
    nslc: str | None = Field(
        default=None, description="Nslc. Package option: __init__."
    )
    training_hash: str | None = Field(
        default=None, description="Training hash. Package option: __init__."
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


class PredictionLabel(Parameters):
    """Parameters for ``model/prediction_model.py:build_label``."""

    window_step: int | None = Field(
        default=None, description="Window step. Package option: build_label."
    )
    window_step_unit: Literal["minutes", "hours"] | None = Field(
        default=None, description="Window step unit. Package option: build_label."
    )


class PredictionFeatures(Parameters):
    """Parameters for ``model/prediction_model.py:extract_features``."""

    select_tremor_columns: list[str] | None = Field(
        default=None,
        description="Select tremor columns. Package option: extract_features.",
    )
    save_tremor_matrix_per_method: bool = Field(
        default=False,
        description="Save tremor matrix per method. Package option: extract_features.",
    )
    exclude_features: list[str] | None = Field(
        default=None, description="Exclude features. Package option: extract_features."
    )
    select_features: str | list[str] | None = Field(
        default=None, description="Select features. Package option: extract_features."
    )
    overwrite: bool | None = Field(
        default=False, description="Overwrite. Package option: extract_features."
    )
    n_jobs: int | None = Field(
        default=None, description="N jobs. Package option: extract_features."
    )
    verbose: bool | None = Field(
        default=None, description="Verbose. Package option: extract_features."
    )


class PredictionForecast(Parameters):
    """Parameters for ``model/prediction_model.py:forecast``."""

    save_seed_result: bool = Field(
        default=True, description="Save seed result. Package option: forecast."
    )
    plot_threshold: float = Field(
        default=0.5, description="Plot threshold. Package option: forecast."
    )
    plot_title: str | None = Field(
        default=None, description="Plot title. Package option: forecast."
    )
    plot_pdf: bool = Field(
        default=True, description="Plot pdf. Package option: forecast."
    )
    plot_start_date: str | None = Field(
        default=None, description="Plot start date. Package option: forecast."
    )
    plot_end_date: str | None = Field(
        default=None, description="Plot end date. Package option: forecast."
    )


class PredictionLoadFeatures(Parameters):
    """Parameters for ``model/prediction_model.py:load_features``."""

    features_matrix_path: str | None = Field(
        default=None, description="Features matrix path. Package option: load_features."
    )
    label_features_csv: str | None = Field(
        default=None, description="Label features csv. Package option: load_features."
    )
    window_step: int | None = Field(
        default=None, description="Window step. Package option: load_features."
    )
    window_step_unit: Literal["minutes", "hours"] | None = Field(
        default=None, description="Window step unit. Package option: load_features."
    )
