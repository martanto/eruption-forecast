"""Tremor time-series visualization with Nature/Science journal styling."""

import os
from typing import Literal
from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    configure_spine,
    apply_nature_style,
)


def plot_tremor(
    df: pd.DataFrame,
    interval: int = 1,
    interval_unit: Literal["hours", "days"] = "hours",
    filename: str | None = None,
    figure_dir: str | None = None,
    title: str | None = None,
    overwrite: bool = True,
    dpi: int = 150,
    selected_columns: list[str] | None = None,
    verbose: bool = False,
) -> None:
    """Plot tremor data as a multi-panel time series with publication-quality styling.

    Creates one subplot per column in the DataFrame (or per selected column),
    with Nature/Science journal formatting, colorblind-safe colors, and
    configurable x-axis tick interval.

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
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.
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
    start_date: pd.Timestamp = df.index[0]
    end_date: pd.Timestamp = df.index[-1]
    n_days = int((end_date - start_date).days)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    if filename is not None:
        filename = Path(filename).stem

    default_filename = f"tremor_{start_date_str}_{end_date_str}"
    default_title = (
        f"{start_date_str}" if n_days == 0 else f"{start_date_str} to {end_date_str}"
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
        else mdates.DayLocator(interval=interval)
    )
    date_formatter = (
        mdates.DateFormatter("%H:%M")
        if interval_unit == "hours"
        else mdates.DateFormatter("%Y-%m-%d")
    )

    columns = selected_columns or df.columns.tolist()
    n_rows = len(columns)

    # Apply Nature/Science styling
    with apply_nature_style():
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=1,
            figsize=(10, 1.5 * n_rows),
            sharex=True,
        )

        # Ensure axs is always iterable
        if n_rows == 1:
            axs = [axs]

        for index, column in enumerate(columns):
            ax = axs[index]

            # Color selection: use Okabe-Ito palette for different column types
            # RSAM columns in blue tones, DSAR in orange tones
            if "rsam" in column.lower():
                color = OKABE_ITO[4]  # Blue
            elif "dsar" in column.lower():
                color = OKABE_ITO[0]  # Orange
            else:
                color = OKABE_ITO[index % len(OKABE_ITO)]

            ax.plot(
                df.index,
                df[column],
                color=color,
                linewidth=1.2,
                label=column.upper(),
                alpha=0.85,
            )
            ax.set_xlim(start_date, end_date)

            # Configure axes
            configure_spine(ax)
            ax.legend(loc="upper left", frameon=False)

            # Add y-axis label with units
            ylabel = "Amplitude (counts)" if "rsam" in column.lower() else "Ratio"
            ax.set_ylabel(ylabel)

            # Configure x-axis
            ax.xaxis.set_major_locator(date_locator)
            ax.xaxis.set_major_formatter(date_formatter)

            # Rotate x-axis labels for better readability
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=30, horizontalalignment="right")

            # Add x-axis label only to bottom subplot
            if index == (n_rows - 1):
                ax.set_xlabel(f"Time ({title})")

        plt.savefig(filepath, dpi=dpi)
        plt.close()

    if verbose:
        logger.info(f"{start_date_str} :: Plot saved to {filepath}")

    return None
