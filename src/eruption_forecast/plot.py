# Standard library imports
import os
from pathlib import Path
from typing import Literal

# Third party imports
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt

# Project imports
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
    """Plot tremor data

    Args:
        df (pd.DataFrame): Tremor data
        interval (Optional[int], optional): Interval in day. Defaults to 1.
        interval_unit (Literal["hours", "days"], optional): Interval unit. Defaults to "hours".
        filename (Optional[str], optional): Filename. Defaults to None.
        figure_dir (Optional[str], optional): Output directory. Defaults to None.
        title (Optional[str], optional): Plot Title. Defaults to None.
        overwrite (bool, optional): Overwrite. Defaults to False.
        dpi (Optional[int], optional): Plot resolution. Defaults to 100.
        selected_columns (Optional[list], optional): Selected columns to plot. Defaults to None.
        verbose (bool, optional): Verbosity. Defaults to False.

    Returns:
        None
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
    values_column: str = "values",
    dpi: int = 100,
    overwrite: bool = True,
):
    """Plot significant features

    Args:
        df (pd.DataFrame): Significant features data
        filepath (str): Save filepath location.
        number_of_features (int, optional): Number of features. Defaults to 50.
        top_features (int, optional): Number of top features. Defaults to 20.
        title (Optional[str], optional): Plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (3, 12).
        features_column (str, optional): Features column name. Defaults to "features".
        values_column (str, optional): Values column name. Defaults to "values".
        dpi (int, optional): DPI. Defaults to 100.
        overwrite (bool, optional): Overwrite. Defaults to True.

    Returns:
        None
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

    df.dropna(inplace=True)
    df = df.head(number_of_features)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.grid(True, linestyle="-.", axis="both", alpha=0.3)
    ax.barh(df[features_column], df[values_column], height=0.5)
    ax.axhline(
        df.index[top_features - 1],
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
