from typing import Literal
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import DIVERGING_BREWER
from eruption_forecast.utils.pathutils import save_figure
from eruption_forecast.config.constants import CALCULATE_METHODS
from eruption_forecast.utils.date_utils import sort_dates, to_datetime
from eruption_forecast.plots.forecast_plots import ax_eruption


def plot_tremor(
    df: pd.DataFrame,
    rolling_window: str = "2D",
    interval: int = 1,
    interval_unit: Literal["hours", "days"] = "hours",
    eruption_dates: list[str] | None = None,
    title: str | None = None,
    selected_columns: list[str] | None = None,
    metrics: Literal["median", "mean", "all"] = "all",
    filepath: str | None = None,
    dpi: int = 150,
    rsam_as_log: bool = False,
    legend_loc: str = "best",
    legend_ncol: int = 2,
    grouped_by_method: bool = True,
    verbose: bool = False,
) -> plt.Figure:
    """Plot tremor time-series grouped by calculation method.

    Renders one subplot per detected method (``rsam``, ``dsar``, ``entropy``)
    using the columns present in ``df``. By default each series is shown as a
    rolling median (solid) and rolling mean (dashed); use ``metrics`` to
    restrict the output to one of the two. Optional eruption markers and a
    figure title can be overlaid. When ``filepath`` is supplied, the figure is
    saved via ``save_figure`` (parent directories are created automatically and
    the extension is appended when missing); otherwise it is returned without
    being written.

    Args:
        df (pd.DataFrame): Tremor DataFrame with a ``DatetimeIndex`` and one
            column per metric (e.g. ``rsam_f0``, ``dsar_f0-f1``, ``entropy``).
        rolling_window (str): Pandas offset alias for the rolling reduction
            (median + mean). Defaults to ``"2D"``.
        interval (int): Tick interval for the x-axis date locator. Defaults to ``1``.
        interval_unit (Literal["hours", "days"]): Unit applied to ``interval``.
            ``"hours"`` uses ``HourLocator`` + ``%H:%M`` formatting; ``"days"``
            uses ``DayLocator`` + ``%Y-%m-%d``. Defaults to ``"hours"``.
        eruption_dates (list[str] | None): Eruption timestamps to overlay as
            vertical markers (one labelled ``"Eruption"`` legend entry per
            subplot). Markers outside the DataFrame range are skipped.
            Defaults to ``None``.
        title (str | None): Optional figure-level ``suptitle``. Defaults to ``None``.
        selected_columns (list[str] | None): Subset of columns to plot. When
            provided, ``df`` is narrowed to these columns before grouping.
            Defaults to ``None`` (all columns).
        metrics (Literal["median", "mean", "all"]): Which rolling reduction(s)
            to draw per series. ``"all"`` (default) draws both the rolling
            median (solid) and rolling mean (dashed); ``"median"`` draws the
            solid median only; ``"mean"`` draws the dashed mean only.
        filepath (str | None): Absolute path where the figure should be saved.
            Parent directories are created by ``save_figure``; the extension is
            appended when missing. Callers own the overwrite policy — passing
            ``filepath`` always (re)writes the file. Defaults to ``None``
            (figure is returned without being saved).
        dpi (int): Resolution used when saving. Ignored if ``filepath`` is
            ``None``. Defaults to ``150``.
        rsam_as_log (bool): If ``True``, plot the RSAM subplot on a log y-axis
            and annotate the y-label with ``"(log)"``. Defaults to ``False``.
        legend_loc (str): Matplotlib legend location string. Defaults to ``"best"``.
        legend_ncol (int): Legend column count. Defaults to ``2``.
        grouped_by_method (bool): If ``True`` (default), group columns by
            tremor method and render one subplot per method, stacking every
            column from the same family (e.g. ``rsam_f0``..``rsam_f4``) on a
            shared axis. If ``False``, render one subplot per column in the
            order they appear in ``df`` (or ``selected_columns`` when
            provided) — useful for inspecting individual bands side by side.
            Defaults to ``True``.
        verbose (bool): If ``True``, ``save_figure`` logs the output path.
            Defaults to ``False``.

    Returns:
        plt.Figure: The matplotlib figure. Always returned, whether or not
        ``filepath`` was supplied.

    Raises:
        TypeError: If ``df.index`` is not a ``pd.DatetimeIndex``.
        ValueError: If ``selected_columns`` cannot be applied to ``df``, or if
            no recognised tremor methods are present in the (possibly narrowed)
            column set.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("tremor.csv", index_col=0, parse_dates=True)
        >>> fig = plot_tremor(
        ...     df=df,
        ...     interval=6,
        ...     interval_unit="hours",
        ...     eruption_dates=["2025-03-20"],
        ...     title="VG.OJN.00.EHZ",
        ...     filepath="output/tremor.png",
        ... )
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Dataframe doesn't have datetime index. Your index type {type(df.index)}"
        )

    if selected_columns:
        try:
            df = df[selected_columns]
        except Exception as e:
            raise ValueError(f"Could not select columns [{selected_columns}]. {e}")

    start_date = df.index.min()
    end_date = df.index.max()

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

    grouped_methods: dict[str, list[str]] = {}
    if grouped_by_method:
        for method in CALCULATE_METHODS:
            cols = sorted(c for c in columns if method in c)
            if cols:
                grouped_methods[method] = cols
    else:
        grouped_methods = {column: [column] for column in columns}

    if not grouped_methods:
        raise ValueError(f"There are no tremor available for columns {columns}")

    nrows = len(grouped_methods)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12, 2.5 * nrows),
        layout="constrained",
        sharex=True,
        squeeze=False,
    )

    for index, (method, columns) in enumerate(grouped_methods.items()):
        if len(columns) == 0:
            continue

        ax = axs[index, 0]
        for column_index, column in enumerate(columns):
            labels = column.split("_")
            label = labels[0]
            if len(labels) == 2:
                label = labels[1]

            if metrics == "all" or metrics == "median":
                ax.plot(
                    df.index,
                    df[column].rolling(window=rolling_window, center=True).median(),
                    color=DIVERGING_BREWER[column_index],
                    alpha=0.8,
                    label=f"2d|median|{label}",
                )

            if metrics == "all" or metrics == "mean":
                ax.plot(
                    df.index,
                    df[column].rolling(window=rolling_window, center=True).mean(),
                    color=DIVERGING_BREWER[column_index],
                    alpha=0.5,
                    label=f"2d|mean|{label}",
                    linestyle="--",
                )

        if eruption_dates is not None and len(eruption_dates) > 0:
            _eruption_dates = sort_dates(eruption_dates, as_datetime=True)

            for _index, eruption_date in enumerate(_eruption_dates):
                label = "Eruption" if _index == (len(_eruption_dates) - 1) else None
                if df.index[0] <= eruption_date <= df.index[-1]:
                    ax = ax_eruption(
                        ax, to_datetime(eruption_date), label=label, fill_between=True
                    )

        # Add y-axis label with units
        if "rsam" in method.lower():
            ylabel = "Amp. "
            if rsam_as_log:
                ylabel = f"{ylabel} (log)"
        elif "dsar" in method.lower():
            ylabel = "Ratio"
        elif "entropy" in method.lower():
            ylabel = "Entropy"
        else:
            ylabel = "A.U."
        ax.set_ylabel(ylabel)

        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlim(start_date, end_date)
        for label in ax.get_xticklabels(which="major"):
            label.set(rotation=30, horizontalalignment="right")

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        ax.legend(
            loc=legend_loc,  # ty:ignore[invalid-argument-type]
            frameon=False,
            ncol=legend_ncol,
            fontsize=8,
        )
        ax.set_title(method.upper(), fontsize=10)

        if "rsam" in method.lower() and rsam_as_log:
            ax.set_yscale("log")

    if title:
        fig.suptitle(title)

    if filepath:
        save_figure(fig, filepath, dpi=dpi, verbose=verbose)

    return fig


def plot_tremor_from_file(
    csv_path: str | Path,
    **kwargs,
) -> plt.Figure:
    """Load a tremor CSV from disk and render it via :func:`plot_tremor`.

    Thin convenience wrapper around :func:`plot_tremor` for the common
    single-file case: read the CSV with a datetime index, then forward the
    resulting DataFrame plus every other keyword to :func:`plot_tremor`. Use
    :func:`replot_tremor` instead when batching a whole directory.

    Args:
        csv_path (str | Path): Path to a tremor CSV produced by
            ``CalculateTremor`` (datetime index in the first column, tremor
            columns such as ``rsam_f0``, ``dsar_f0-f1``, ``entropy``).
        **kwargs: Additional keyword arguments forwarded verbatim to
            :func:`plot_tremor` (e.g. ``rolling_window``, ``interval``,
            ``interval_unit``, ``eruption_dates``, ``title``,
            ``selected_columns``, ``metrics``, ``filepath``, ``dpi``,
            ``rsam_as_log``, ``legend_loc``, ``legend_ncol``,
            ``grouped_by_method``, ``verbose``). ``df`` is set internally
            from ``csv_path``; passing it here would raise ``TypeError``.

    Returns:
        plt.Figure: The figure returned by :func:`plot_tremor`. Saved to disk
        only when ``filepath`` is supplied via ``kwargs``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.

    Example:
        >>> fig = plot_tremor_from_file(
        ...     "output/VG.OJN.00.EHZ/tremor/daily/2025-03-20.csv",
        ...     interval=6,
        ...     interval_unit="hours",
        ...     eruption_dates=["2025-03-20"],
        ...     title="VG.OJN.00.EHZ — 2025-03-20",
        ...     filepath="tremor_2025-03-20.png",
        ... )
    """
    df = pd.read_csv(Path(csv_path), index_col=0, parse_dates=True)
    return plot_tremor(df=df, **kwargs)


def _process_single_tremor_file(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool,
    plot_tremor_kwargs: dict,
) -> str:
    """Process a single tremor CSV file and generate a plot.

    Helper for batch processing in ``replot_tremor()``. Loads a tremor CSV
    file, derives the output PNG path as
    ``output_dir / f"{csv_path.stem}.png"``, calls ``plot_tremor()`` with that
    full path, and returns a status string. Designed for use with
    ``multiprocessing.Pool.starmap()``.

    Args:
        csv_path (Path): Path to the CSV file containing tremor data. Must
            have a datetime index and tremor columns (e.g. ``rsam_f0``,
            ``dsar_f0-f1``).
        output_dir (Path): Directory where the plot PNG will be written. The
            file is named after ``csv_path.stem``.
        overwrite (bool): If ``True``, regenerate the plot even when the
            target PNG already exists. If ``False`` and the PNG exists, the
            file is left untouched and ``"skipped"`` is returned.
        plot_tremor_kwargs (dict): Additional keyword arguments forwarded to
            ``plot_tremor()``. Typical keys: ``interval``, ``interval_unit``,
            ``selected_columns``, ``title``, ``dpi``, ``verbose``,
            ``rsam_as_log``, ``legend_loc``. **Do not include ``filepath``** —
            this helper sets it from ``output_dir`` / ``csv_path.stem`` and a
            duplicate would raise ``TypeError``.

    Returns:
        str: Processing status — one of ``"created"``, ``"skipped"``, or
        ``"failed"``.

    Notes:
        Errors are caught and logged via ``logger.error()``; the function
        never raises and instead returns ``"failed"`` so a failing CSV does
        not abort the surrounding batch.
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

        plot_tremor(
            df=df,
            filepath=str(output_path),
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
        **kwargs: Additional keyword arguments forwarded to ``plot_tremor()``.
            Typical keys: ``interval``, ``interval_unit``, ``selected_columns``,
            ``title``, ``dpi``, ``verbose``, ``rsam_as_log``, ``legend_loc``.
            **Do not include ``filepath``** — it is set internally per CSV from
            ``output_dir`` and the CSV stem; passing it here would raise
            ``TypeError``.

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
        ...     selected_columns=["rsam_f0", "rsam_f1"],
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
    job_results = []
    if n_jobs == 1:
        # Sequential processing
        job_results = [_process_single_tremor_file(*job) for job in jobs]  # type: ignore[arg-type]
    else:
        # Parallel processing
        logger.info(f"Running on {n_jobs} job(s)")
        with Pool(n_jobs) as pool:
            job_results = pool.starmap(_process_single_tremor_file, jobs)

    if len(job_results) == 0:
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
