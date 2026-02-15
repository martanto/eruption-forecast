"""Legacy plotting functions - delegates to eruption_forecast.plots module.

This module maintains backward compatibility by wrapping the new Nature/Science
styled plotting functions. All existing imports continue to work without changes.

For new code, prefer importing directly from eruption_forecast.plots submodules.
"""

from typing import Literal

import pandas as pd

from eruption_forecast.plots.tremor_plots import plot_tremor as _plot_tremor_new
from eruption_forecast.plots.feature_plots import (
    plot_significant_features as _plot_significant_features_new,
)


def plot_tremor(
    df: pd.DataFrame,
    interval: int = 1,
    interval_unit: Literal["hours", "days"] = "hours",
    filename: str | None = None,
    figure_dir: str | None = None,
    title: str | None = None,
    overwrite: bool = True,
    dpi: int = 100,
    selected_columns: list[str] | None = None,
    verbose: bool = False,
) -> None:
    """Plot tremor data as a multi-panel time series figure.

    **Legacy wrapper**: Delegates to eruption_forecast.plots.tremor_plots.plot_tremor
    with Nature/Science journal styling.

    Creates one subplot per column in the DataFrame (or per selected column),
    with configurable x-axis tick interval and optional file output.

    Args:
        df (pd.DataFrame): Tremor data with a DatetimeIndex.
        interval (int, optional): Tick interval for the x-axis. Defaults to 1.
        interval_unit (Literal["hours", "days"], optional): Unit for the tick
            interval — ``"hours"`` or ``"days"``. Defaults to ``"hours"``.
        filename (str | None, optional): Output filename stem (extension is
            added automatically). If None, a name is derived from the date
            range. Defaults to None.
        figure_dir (str | None, optional): Directory to save the figure. If
            None, saves to ``<cwd>/figures``. Defaults to None.
        title (str | None, optional): X-axis label / plot title. If None, the
            date range is used. Defaults to None.
        overwrite (bool, optional): If True, overwrite an existing file with
            the same name. Defaults to True.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 100.
        selected_columns (list[str] | None, optional): Subset of columns to
            plot. If None, all columns are plotted. Defaults to None.
        verbose (bool, optional): If True, log a message when the file is
            saved or already exists. Defaults to False.

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> plot_tremor(df, interval=6, interval_unit="hours",
        ...            figure_dir="output/figures", overwrite=False)
    """
    # Use enhanced DPI if not explicitly overridden
    if dpi == 100:
        dpi = 150

    return _plot_tremor_new(
        df=df,
        interval=interval,
        interval_unit=interval_unit,
        filename=filename,
        figure_dir=figure_dir,
        title=title,
        overwrite=overwrite,
        dpi=dpi,
        selected_columns=selected_columns,
        verbose=verbose,
    )


def plot_significant_features(
    df: pd.DataFrame,
    filepath: str,
    number_of_features: int = 50,
    top_features: int = 20,
    title: str | None = None,
    figsize=(3, 12),
    features_column: str = "features",
    values_column: str = "p_values",
    dpi: int = 100,
    overwrite: bool = True,
):
    """Plot a horizontal bar chart of significant features.

    **Legacy wrapper**: Delegates to eruption_forecast.plots.feature_plots.plot_significant_features
    with Nature/Science journal styling.

    Displays the top ``number_of_features`` rows of ``df`` as a horizontal
    bar chart sorted by ``values_column``, with a dashed reference line at
    ``top_features``.

    Args:
        df (pd.DataFrame): DataFrame containing feature names and their
            significance values (e.g. p-values).
        filepath (str): Full path (including filename) where the figure is
            saved.
        number_of_features (int, optional): Total number of features to
            display in the chart. Defaults to 50.
        top_features (int, optional): Position at which to draw a reference
            line marking the top-N cut-off. Defaults to 20.
        title (str | None, optional): Chart title. If None, defaults to
            ``"<number_of_features> Significant Features"``. Defaults to None.
        figsize (tuple, optional): Figure dimensions as ``(width, height)``
            in inches. Defaults to ``(3, 12)``.
        features_column (str, optional): Name of the column containing
            feature names. If missing, the index is used. Defaults to
            ``"features"``.
        values_column (str, optional): Name of the column containing
            significance values. Defaults to ``"p_values"``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 100.
        overwrite (bool, optional): If True, overwrite an existing file.
            Defaults to True.

    Returns:
        None

    Examples:
        >>> plot_significant_features(
        ...     df=sig_features_df,
        ...     filepath="output/figures/significant_features.png",
        ...     number_of_features=30,
        ...     top_features=10,
        ... )
    """
    # Use enhanced DPI if not explicitly overridden
    if dpi == 100:
        dpi = 150

    # Convert tuple to explicit tuple type if needed
    if not isinstance(figsize, tuple):
        figsize = tuple(figsize)

    return _plot_significant_features_new(
        df=df,
        filepath=filepath,
        number_of_features=number_of_features,
        top_features=top_features,
        title=title,
        figsize=figsize,
        features_column=features_column,
        values_column=values_column,
        dpi=dpi,
        overwrite=overwrite,
    )

