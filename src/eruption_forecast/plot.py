import os
from typing import Literal
from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

from eruption_forecast.logger import logger


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
    # TODO: Asserting pd.datetime index
    # TODO: Asserting interval_unit
    # TODO: Asserting selected_columns

    start_date: pd.Timestamp = df.index[0]
    end_date: pd.Timestamp = df.index[-1]
    n_days = int((end_date - start_date).days)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    if filename is not None:
        filename = Path(filename).stem

    default_filename = f"tremor_{start_date_str}_{end_date_str}"
    default_title = (
        f"{start_date_str}" if n_days == 0 else f"{start_date_str}_{end_date_str}"
    )
    title = title or default_title

    # Save plot to figure directory
    figure_dir = figure_dir or os.path.join(os.getcwd(), "figures")
    os.makedirs(figure_dir, exist_ok=True)

    filename = filename or default_filename
    filepath = os.path.join(figure_dir, f"{filename}.png")

    if os.path.exists(filepath) and not overwrite:
        if verbose:
            logger.info(f"{start_date_str} :: Plot already exists at {filepath}")
        return None

    # Define date locator and formatter based on plot type
    date_locator = (
        mdates.HourLocator(interval=interval)
        if interval_unit == "hours"
        else mdates.DayLocator(interval=14)
    )
    date_formatter = (
        mdates.DateFormatter("%H:%M")
        if interval_unit == "hours"
        else mdates.DateFormatter("%Y-%m-%d")
    )

    columns = selected_columns or df.columns.tolist()
    n_rows = len(columns)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=1, figsize=(10, 1.2 * n_rows), sharex=True
    )

    for index, column in enumerate(columns):
        ax = axs[index] if n_rows > 1 else axs
        ax.grid(True, linestyle="-.", axis="both", alpha=0.5)
        ax.plot(
            df.index,
            df[column],
            color="black",
            linewidth=1,
            label=column,
            alpha=0.8,
        )
        ax.set_xlim(start_date, end_date)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        for label in ax.get_xticklabels(which="major"):
            label.set(rotation=30, horizontalalignment="right", fontsize=8)

        if index == (n_rows - 1):
            ax.set_xlabel(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi)
    plt.close()

    if verbose:
        logger.info(f"{start_date_str} :: Plot saved to {filepath}")

    return None


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
    if (filepath is not None) and (not overwrite) and os.path.isfile(filepath):
        return None

    if features_column not in df.columns:
        try:
            df[features_column] = df.index
        except ValueError:
            raise ValueError(  # noqa: B904
                f"Features column: {features_column} does not exist"
            )  # noqa: B904

    df = df.dropna()
    df = df.head(number_of_features)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.grid(True, linestyle="-.", axis="both", alpha=0.3)
    ax.barh(df[features_column], df[values_column], height=0.5)
    ax.axhline(
        df.index[top_features],
        color="red",
        linestyle="--",
        label=f"Top {top_features} Fts",
    )

    ax.title.set_text(title or f"{number_of_features} Significant Features")

    ax.legend(frameon=False)

    plt.ylim(-0.5, number_of_features - 0.5)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()

    del df

    return None
