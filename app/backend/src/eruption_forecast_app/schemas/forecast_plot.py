"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class ForecastPlot(Parameters):
    """Parameters for ``plots/forecast_plots.py:plot_forecast``."""

    title: str | None = Field(
        default=None, description="Title. Package option: plot_forecast."
    )
    fig_width: float = Field(
        default=8, description="Fig width. Package option: plot_forecast."
    )
    fig_height: float = Field(
        default=1.5, description="Fig height. Package option: plot_forecast."
    )
    threshold: float = Field(
        default=0.7, description="Threshold. Package option: plot_forecast."
    )
    rolling_window: str = Field(
        default="6h", description="Rolling window. Package option: plot_forecast."
    )
    x_days_interval: int = Field(
        default=2, description="X days interval. Package option: plot_forecast."
    )
    x_label_fontsize: int = Field(
        default=7, description="X label fontsize. Package option: plot_forecast."
    )
    x_label_rotation: int = Field(
        default=-15, description="X label rotation. Package option: plot_forecast."
    )
    eruption_dates: list[str] | None = Field(
        default=None, description="Eruption dates. Package option: plot_forecast."
    )
    y_max: float = Field(
        default=1.05, description="Y max. Package option: plot_forecast."
    )
    legend_n_cols: int = Field(
        default=6, description="Legend n cols. Package option: plot_forecast."
    )
    legend_fontsize: int = Field(
        default=6, description="Legend fontsize. Package option: plot_forecast."
    )
    legend_position: tuple[float, float] = Field(
        default=(0.5, -0.075),
        description="Legend position. Package option: plot_forecast.",
    )
    training_start_date: str | None = Field(
        default=None, description="Training start date. Package option: plot_forecast."
    )
    training_end_date: str | None = Field(
        default=None, description="Training end date. Package option: plot_forecast."
    )
    prediction_start_date: str | None = Field(
        default=None,
        description="Prediction start date. Package option: plot_forecast.",
    )
    prediction_end_date: str | None = Field(
        default=None, description="Prediction end date. Package option: plot_forecast."
    )
    start_date: str | None = Field(
        default=None, description="Start date. Package option: plot_forecast."
    )
    end_date: str | None = Field(
        default=None, description="End date. Package option: plot_forecast."
    )
