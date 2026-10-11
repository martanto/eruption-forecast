"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from typing import Literal

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class TrainingConstructor(Parameters):
    """Parameters for ``model/training_model.py:__init__``."""

    tremor_data: str | None = Field(
        default=None, description="Tremor data. Package option: __init__."
    )
    start_date: str | None = Field(
        default=None, description="Start date. Package option: __init__."
    )
    end_date: str | None = Field(
        default=None, description="End date. Package option: __init__."
    )
    classifiers: str | list[str] | None = Field(
        default=None, description="Classifiers. Package option: __init__."
    )
    eruption_dates: list[str] | None = Field(
        default=None, description="Eruption dates. Package option: __init__."
    )
    window_size: int = Field(
        default=2, description="Window size. Package option: __init__."
    )
    cv_strategy: Literal["shuffle", "stratified", "shuffle-stratified"] = Field(
        default="shuffle-stratified",
        description="Cv strategy. Package option: __init__.",
    )
    cv_splits: int = Field(
        default=5, description="Cv splits. Package option: __init__."
    )
    top_n_features: int = Field(
        default=20, description="Top n features. Package option: __init__."
    )
    include_eruption_date: bool = Field(
        default=False, description="Include eruption date. Package option: __init__."
    )
    nslc: str | None = Field(
        default=None, description="Nslc. Package option: __init__."
    )
    output_dir: str | None = Field(
        default=None, description="Output dir. Package option: __init__."
    )
    root_dir: str | None = Field(
        default=None, description="Root dir. Package option: __init__."
    )
    overwrite: bool | None = Field(
        default=False, description="Overwrite. Package option: __init__."
    )
    prefix_config: str | None = Field(
        default=None, description="Prefix config. Package option: __init__."
    )
    n_jobs: int | None = Field(
        default=1, description="N jobs. Package option: __init__."
    )
    n_grids: int = Field(default=1, description="N grids. Package option: __init__.")
    save_model: bool = Field(
        default=False, description="Save model. Package option: __init__."
    )
    verbose: bool | None = Field(
        default=False, description="Verbose. Package option: __init__."
    )


class TrainingLabel(Parameters):
    """Parameters for ``model/training_model.py:build_label``."""

    window_step: int | None = Field(
        default=None, description="Window step. Package option: build_label."
    )
    window_step_unit: Literal["minutes", "hours"] | None = Field(
        default=None, description="Window step unit. Package option: build_label."
    )
    builder: Literal["standard", "dynamic"] = Field(
        default="standard", description="Builder. Package option: build_label."
    )
    days_before_eruption: int | None = Field(
        default=None, description="Days before eruption. Package option: build_label."
    )
    verbose: bool | None = Field(
        default=None, description="Verbose. Package option: build_label."
    )


class TrainingFeatures(Parameters):
    """Parameters for ``model/training_model.py:extract_features``."""

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
    save_tremor_matrix_per_id: bool = Field(
        default=False,
        description="Save tremor matrix per id. Package option: extract_features.",
    )
    minimum_completion: float = Field(
        default=1.0, description="Minimum completion. Package option: extract_features."
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


class TrainingFit(Parameters):
    """Parameters for ``model/training_model.py:fit``."""

    seeds: int = Field(default=25, description="Seeds. Package option: fit.")
    resample_method: Literal["under", "over", "auto"] | None = Field(
        default="auto", description="Resample method. Package option: fit."
    )
    minority_threshold: float = Field(
        default=0.15, description="Minority threshold. Package option: fit."
    )
    sampling_strategy: str | float = Field(
        default=0.75, description="Sampling strategy. Package option: fit."
    )
    plot_features: bool = Field(
        default=False, description="Plot features. Package option: fit."
    )
    scoring: str = Field(
        default="balanced_accuracy", description="Scoring. Package option: fit."
    )
    compute_learning_curve: bool = Field(
        default=False, description="Compute learning curve. Package option: fit."
    )


class TrainingLoadFeatures(Parameters):
    """Parameters for ``model/training_model.py:load_features``."""

    features_matrix_path: str | None = Field(
        default=None, description="Features matrix path. Package option: load_features."
    )
    label_features_csv: str | None = Field(
        default=None, description="Label features csv. Package option: load_features."
    )
    select_features: str | list[str] | None = Field(
        default=None, description="Select features. Package option: load_features."
    )
