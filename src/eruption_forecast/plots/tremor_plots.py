"""Tremor time-series visualization with Nature/Science journal styling."""

import os
from typing import Literal
from pathlib import Path
from multiprocessing import Pool

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


def _process_single_tremor_file(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool,
    plot_tremor_kwargs: dict,
) -> str:
    """Process a single tremor CSV file and generate a plot.

    Helper function for batch processing in replot_tremor(). Loads a tremor
    CSV file, generates a plot using plot_tremor(), and returns the processing
    status.

    Args:
        csv_path (Path): Path to the CSV file containing tremor data.
        output_dir (Path): Directory where the plot will be saved.
        overwrite (bool): If True, overwrite existing plots. If False, skip
            if plot already exists.
        plot_tremor_kwargs (dict): Additional keyword arguments to pass to
            plot_tremor().

    Returns:
        str: Processing status - "created", "skipped", or "failed".

    Notes:
        - CSV file is expected to have a datetime index and tremor columns
        - Filename stem is used as the plot filename
        - Errors are logged but don't raise exceptions
    """
    try:
        # Load CSV with datetime index
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Extract filename stem for plot naming
        filename_stem = csv_path.stem

        # Check if plot already exists and overwrite is False
        output_path = output_dir / f"{filename_stem}.png"
        if output_path.exists() and not overwrite:
            return "skipped"

        # Generate plot using plot_tremor
        plot_tremor(
            df=df,
            filename=filename_stem,
            figure_dir=str(output_dir),
            overwrite=overwrite,
            **plot_tremor_kwargs,
        )

        return "created"

    except Exception as e:
        logger.error(f"Failed to plot {csv_path.name}: {e}")
        return "failed"


def replot_tremor(
    daily_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
    n_jobs: int = 1,
    **kwargs,
) -> dict[str, int]:
    """Batch replot daily tremor data from all CSV files in a directory.

    Reads all CSV files from the specified directory, loads each as a DataFrame,
    and generates publication-quality tremor time-series plots using
    ``plot_tremor()``. Useful for replotting daily tremor calculations from
    CalculateTremor with consistent styling.

    Args:
        daily_dir (str | Path): Directory containing daily tremor CSV files.
            Each CSV should have a datetime index and tremor columns (e.g.,
            rsam_f0, rsam_f1, dsar_f0-f1). Typically the ``daily/`` subdirectory
            created by CalculateTremor. **REQUIRED parameter.**
        output_dir (str | Path | None, optional): Directory where output plots
            will be saved. If None, plots are saved in ``<parent>/figures``
            where ``<parent>`` is the parent directory of ``daily_dir``. For
            example, if input is ``.../tremor/daily``, output will be
            ``.../tremor/figures``. Defaults to None.
        overwrite (bool, optional): If True, regenerate all plots. If False,
            skip plotting if the output file already exists. Defaults to True.
        n_jobs (int, optional): Number of parallel jobs for plotting. If 1,
            processes files sequentially. If greater than 1, uses multiprocessing
            to plot multiple files in parallel. Defaults to 1.
        **kwargs: Additional keyword arguments passed to ``plot_tremor()``.
            Can include ``interval``, ``interval_unit``, ``selected_columns``,
            ``title``, ``dpi``, ``verbose``, etc.

    Returns:
        dict[str, int]: Summary statistics with keys:
            - ``'created'``: Number of plots successfully created
            - ``'skipped'``: Number of plots skipped (file exists, overwrite=False)
            - ``'failed'``: Number of plots that failed due to errors

    Raises:
        FileNotFoundError: If ``daily_dir`` does not exist.
        NotADirectoryError: If ``daily_dir`` is not a directory.
        ValueError: If ``n_jobs`` is less than or equal to 0.

    Examples:
        >>> # Replot all daily tremor files with default settings
        >>> results = replot_tremor(
        ...     daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
        ...     overwrite=True,
        ... )
        >>> print(f"Created: {results['created']}, Failed: {results['failed']}")

        >>> # Custom output directory, skip existing plots
        >>> results = replot_tremor(
        ...     daily_dir="path/to/tremor/daily",
        ...     output_dir="path/to/figures",
        ...     overwrite=False,
        ... )

        >>> # Parallel processing with custom plot parameters
        >>> results = replot_tremor(
        ...     daily_dir="path/to/tremor/daily",
        ...     n_jobs=4,
        ...     interval=6,
        ...     interval_unit="hours",
        ...     dpi=300,
        ... )

    Notes:
        - CSV files are expected to have a datetime index and tremor columns
        - Output filenames match input CSV filenames with .png extension
        - Errors are logged but don't stop processing of remaining files
        - Default output directory is ``<parent>/figures`` (sibling to daily_dir)
        - Uses the same Nature/Science styling as plot_tremor()
        - When n_jobs > 1, uses multiprocessing for faster batch processing
    """
    # Convert paths to Path objects
    daily_dir = Path(daily_dir)
    if output_dir is None:
        # Default: create sibling directory called 'figures'
        # Example: .../tremor/daily -> .../tremor/figures
        output_dir = daily_dir.parent / "figures"
    else:
        output_dir = Path(output_dir)

    # Validate input directory
    if not daily_dir.exists():
        msg = f"Directory does not exist: {daily_dir}"
        raise FileNotFoundError(msg)
    if not daily_dir.is_dir():
        msg = f"Path is not a directory: {daily_dir}"
        raise NotADirectoryError(msg)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(daily_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {daily_dir}")
        return {"created": 0, "skipped": 0, "failed": 0}

    logger.info(f"Found {len(csv_files)} CSV files in {daily_dir}")

    # Validate n_jobs
    if n_jobs <= 0:
        raise ValueError(f"n_jobs must be greater than 0. Your value: {n_jobs}")

    # Prepare job parameters
    jobs = [
        (
            csv_path,
            output_dir,
            overwrite,
            kwargs,
        )
        for csv_path in csv_files
    ]

    # Process files (sequential or parallel)
    job_results = None
    if n_jobs == 1:
        # Sequential processing
        job_results = [_process_single_tremor_file(*job) for job in jobs]  # type: ignore[arg-type]
    else:
        # Parallel processing
        logger.info(f"Running on {n_jobs} job(s)")
        with Pool(n_jobs) as pool:
            job_results = pool.starmap(_process_single_tremor_file, jobs)

    if job_results is None:
        raise ValueError(f"No results from {n_jobs} job(s)")

    # Aggregate results
    results = {"created": 0, "skipped": 0, "failed": 0}
    for result in job_results:
        results[result] += 1

    # Summary
    logger.info(
        f"Batch replot complete: {results['created']} created, "
        f"{results['skipped']} skipped, {results['failed']} failed"
    )

    return results
