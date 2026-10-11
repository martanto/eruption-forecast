"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from typing import Literal

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class CalculationConstructor(Parameters):
    """Parameters for ``tremor/calculate_tremor.py:__init__``."""

    start_date: str | None = Field(
        default=None, description="Start date. Package option: __init__."
    )
    end_date: str | None = Field(
        default=None, description="End date. Package option: __init__."
    )
    station: str | None = Field(
        default=None, description="Station. Package option: __init__."
    )
    channel: str | None = Field(
        default=None, description="Channel. Package option: __init__."
    )
    network: str | None = Field(
        default=None, description="Network. Package option: __init__."
    )
    location: str | None = Field(
        default=None, description="Location. Package option: __init__."
    )
    channel_type: str = Field(
        default="D", description="Channel type. Package option: __init__."
    )
    methods: list[str] | None = Field(
        default=None, description="Methods. Package option: __init__."
    )
    buffer_days: int | None = Field(
        default=None, description="Buffer days. Package option: __init__."
    )
    remove_outlier_method: Literal["all", "maximum"] = Field(
        default="maximum",
        description="Remove outlier method. Package option: __init__.",
    )
    remove_tremor_anomalies: bool = Field(
        default=False, description="Remove tremor anomalies. Package option: __init__."
    )
    interpolate: bool = Field(
        default=False, description="Interpolate. Package option: __init__."
    )
    value_multiplier: float | None = Field(
        default=None, description="Value multiplier. Package option: __init__."
    )
    cleanup_daily_dir: bool = Field(
        default=False, description="Cleanup daily dir. Package option: __init__."
    )
    plot_daily: bool = Field(
        default=False, description="Plot daily. Package option: __init__."
    )
    save_plot: bool = Field(
        default=False, description="Save plot. Package option: __init__."
    )
    plot_overwrite: bool = Field(
        default=False, description="Plot overwrite. Package option: __init__."
    )
    filename_prefix: str | None = Field(
        default=None, description="Filename prefix. Package option: __init__."
    )
    minimum_completion_ratio: float = Field(
        default=0.3, description="Minimum completion ratio. Package option: __init__."
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
    n_jobs: int | None = Field(
        default=1, description="N jobs. Package option: __init__."
    )
    verbose: bool | None = Field(
        default=False, description="Verbose. Package option: __init__."
    )
    debug: bool = Field(default=False, description="Debug. Package option: __init__.")


class CalculationRun(Parameters):
    """Parameters for ``tremor/calculate_tremor.py:run``."""

    plot_eruption_dates: list[str] | None = Field(
        default=None, description="Plot eruption dates. Package option: run."
    )
    plot_rsam_as_log: bool = Field(
        default=False, description="Plot rsam as log. Package option: run."
    )
    plot_rolling_window: str | None = Field(
        default=None, description="Plot rolling window. Package option: run."
    )
    plot_filter_dsar_value: float | None = Field(
        default=None, description="Plot filter dsar value. Package option: run."
    )
    plot_interval: int = Field(
        default=2, description="Plot interval. Package option: run."
    )
