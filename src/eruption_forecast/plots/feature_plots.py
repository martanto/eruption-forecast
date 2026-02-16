"""Feature importance and selection visualization with Nature/Science styling."""

import os
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    NATURE_COLORS,
    configure_spine,
    apply_nature_style,
)


def plot_significant_features(
    df: pd.DataFrame,
    filepath: str,
    number_of_features: int = 50,
    top_features: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] = (4, 12),
    features_column: str = "features",
    values_column: str = "p_values",
    dpi: int = 150,
    overwrite: bool = True,
) -> None:
    """Plot a horizontal bar chart of significant features with publication-quality styling.

    Displays the top ``number_of_features`` rows of ``df`` as a horizontal
    bar chart sorted by ``values_column``, with a dashed reference line at
    ``top_features``. Uses Nature/Science journal styling with colorblind-safe
    color palette.

    Args:
        df (pd.DataFrame): DataFrame containing feature names and their
            significance values (e.g. p-values or importance scores).
        filepath (str): Full path (including filename) where the figure is
            saved.
        number_of_features (int, optional): Total number of features to
            display in the chart. Defaults to 50.
        top_features (int, optional): Position at which to draw a reference
            line marking the top-N cut-off. Defaults to 20.
        title (str | None, optional): Chart title. If None, defaults to
            ``"<number_of_features> Significant Features"``. Defaults to None.
        figsize (tuple[float, float], optional): Figure dimensions as
            ``(width, height)`` in inches. If using the default value of
            ``(4, 12)``, the height will be automatically calculated based on
            ``number_of_features`` to prevent layout collapse
            (formula: ``max(8, number_of_features * 0.3 + 2)``).
            Custom values are respected as-is. Defaults to ``(4, 12)``.
        features_column (str, optional): Name of the column containing
            feature names. If missing, the index is used. Defaults to
            ``"features"``.
        values_column (str, optional): Name of the column containing
            significance values. Defaults to ``"p_values"``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.
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

    # Prepare dataframe
    if features_column not in df.columns:
        try:
            df = df.copy()
            df[features_column] = df.index
        except ValueError as e:
            msg = f"Features column: {features_column} does not exist"
            raise ValueError(msg) from e

    df = df.dropna()
    df = df.head(number_of_features)
    df = df.iloc[::-1]  # Reverse for bottom-to-top ordering

    # Calculate dynamic figure height based on number of features
    # Only apply if user is using default figsize to avoid breaking custom sizes
    if figsize == (4, 12):  # Default value
        # Formula: min 8" height, or 0.3" per feature + 2" overhead for labels/title
        calculated_height = max(8, number_of_features * 0.3 + 2)
        figsize = (4, calculated_height)

    # Apply Nature/Science styling
    with apply_nature_style():
        # Temporarily disable constrained_layout for horizontal bar charts
        # Use tight_layout instead which handles many labels better
        plt.rcParams["figure.constrained_layout.use"] = False

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # Color bars by position: top features in darker blue
        bar_colors = [
            NATURE_COLORS["blue"]
            if i >= (number_of_features - top_features)
            else NATURE_COLORS["gray"]
            for i in range(len(df))
        ]

        # Create horizontal bar chart
        ax.barh(
            df[features_column],
            df[values_column],
            height=0.6,
            color=bar_colors,
            alpha=0.8,
        )

        # Add reference line for top-N cutoff
        if top_features < number_of_features:
            cutoff_index = number_of_features - top_features - 0.5
            ax.axhline(
                cutoff_index,
                color=NATURE_COLORS["red"],
                linestyle="--",
                linewidth=1.5,
                label=f"Top {top_features} features",
                alpha=0.7,
            )
            ax.legend(frameon=False, loc="upper right")

        # Configure axes
        configure_spine(ax)
        ax.set_xlabel("P-value" if values_column == "p_values" else "Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"{number_of_features} Significant Features")

        # Set y-axis limits
        ax.set_ylim(-0.5, number_of_features - 0.5)

        # Add value labels for top features (optional, for clarity)
        if number_of_features <= 30:  # Only for smaller plots
            for i, (_idx, row) in enumerate(df.iterrows()):
                if i >= (number_of_features - top_features):
                    value = row[values_column]
                    ax.text(
                        value,
                        i,
                        f"  {value:.3f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color=NATURE_COLORS["blue"],
                    )

        # Note: tight_layout() is not called here because savefig.bbox='tight'
        # (configured in styles.py) handles layout automatically and is more
        # robust with long feature labels. Explicit tight_layout() can fail
        # when labels are too long for the figure width.
        plt.savefig(filepath, dpi=dpi)
        plt.close()

    return None


def _process_single_file(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool,
    number_of_features: int,
    top_features: int,
    dpi: int,
    kwargs: dict,
) -> str:
    """Process a single CSV file and generate plot.

    Helper function for multiprocessing support in replot_significant_features().

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory for output plot
        overwrite: Whether to overwrite existing files
        number_of_features: Number of features to display
        top_features: Number of top features to highlight
        dpi: Plot resolution
        kwargs: Additional keyword arguments for plot_significant_features()

    Returns:
        str: Status string - "created", "skipped", or "failed"
    """
    # Generate output filename
    output_filename = csv_path.stem + ".png"
    output_path = output_dir / output_filename

    # Check if should skip
    if not overwrite and output_path.exists():
        logger.debug(f"Skipping {csv_path.name} (already exists)")
        return "skipped"

    try:
        # Load CSV
        df = pd.read_csv(csv_path)

        # Auto-detect features column if not in kwargs
        features_column = kwargs.get("features_column", "features")
        if features_column not in df.columns and len(df.columns) > 0:
            # Assume index contains features
            df = df.copy()
            df[features_column] = df.index
            kwargs["features_column"] = features_column

        # Auto-detect values column if not in kwargs
        values_column = kwargs.get("values_column", None)
        if values_column is None:
            if "p_values" in df.columns:
                values_column = "p_values"
            elif "importance" in df.columns:
                values_column = "importance"
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    values_column = numeric_cols[0]
                else:
                    msg = f"No numeric columns found in {csv_path.name}"
                    raise ValueError(msg)
            kwargs["values_column"] = values_column

        # Plot
        plot_significant_features(
            df=df,
            filepath=str(output_path),
            number_of_features=number_of_features,
            top_features=top_features,
            dpi=dpi,
            overwrite=overwrite,
            **kwargs,
        )

        logger.info(f"Created plot: {output_path.name}")
        return "created"

    except Exception as e:
        logger.error(f"Failed to plot {csv_path.name}: {e}")
        return "failed"


def replot_significant_features(
    all_features_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
    number_of_features: int = 50,
    top_features: int = 20,
    dpi: int = 150,
    n_jobs: int = 1,
    **kwargs,
) -> dict[str, int]:
    """Batch replot significant features from all CSV files in a directory.

    Reads all CSV files from the specified directory, loads each as a DataFrame,
    and generates publication-quality feature importance plots using
    ``plot_significant_features()``. Useful for replotting features across
    multiple random seeds or cross-validation folds.

    Args:
        all_features_dir (str | Path): Directory containing CSV files with
            feature data. Each CSV should have feature names and significance
            values (e.g., p-values or importance scores). **REQUIRED parameter.**
        output_dir (str | Path | None, optional): Directory where output plots
            will be saved. If None, plots are saved in
            ``<parent>/figures/significant`` where ``<parent>`` is the parent
            directory of ``all_features_dir``. For example, if input is
            ``.../features/all_features``, output will be
            ``.../features/figures/significant``. Defaults to None.
        overwrite (bool, optional): If True, regenerate all plots. If False,
            skip plotting if the output file already exists. Defaults to True.
        number_of_features (int, optional): Number of top features to display
            in each plot. Passed to ``plot_significant_features()``.
            Defaults to 50.
        top_features (int, optional): Number of top features to highlight with
            darker color. Passed to ``plot_significant_features()``.
            Defaults to 20.
        dpi (int, optional): Resolution of output plots in dots per inch.
            Passed to ``plot_significant_features()``. Defaults to 150.
        n_jobs (int, optional): Number of parallel jobs for plotting. If 1,
            processes files sequentially. If greater than 1, uses multiprocessing
            to plot multiple files in parallel. Defaults to 1.
        **kwargs: Additional keyword arguments passed to
            ``plot_significant_features()``. Can include ``features_column``,
            ``values_column``, ``title``, ``figsize``, etc.

    Returns:
        dict[str, int]: Summary statistics with keys:
            - ``'created'``: Number of plots successfully created
            - ``'skipped'``: Number of plots skipped (file exists, overwrite=False)
            - ``'failed'``: Number of plots that failed due to errors

    Examples:
        >>> # Replot all features with default settings
        >>> results = replot_significant_features(
        ...     all_features_dir="output/.../features/all_features",
        ...     overwrite=True,
        ... )
        >>> print(f"Created: {results['created']}, Failed: {results['failed']}")

        >>> # Custom output directory, skip existing plots
        >>> results = replot_significant_features(
        ...     all_features_dir="path/to/features",
        ...     output_dir="path/to/plots",
        ...     overwrite=False,
        ...     number_of_features=30,
        ... )

    Notes:
        - CSV files are expected to have either a 'features' column or feature
          names in the index.
        - The function attempts to auto-detect the values column (tries
          'p_values', 'importance', or first numeric column).
        - Errors are logged but don't stop processing of remaining files.
        - Output filenames match input CSV filenames with .png extension.
        - Default output directory is ``<parent>/figures/significant`` where
          ``<parent>`` is the parent directory of ``all_features_dir``.
    """
    # Convert paths to Path objects
    all_features_dir = Path(all_features_dir)
    if output_dir is None:
        # Default: create sibling directory called 'figures/significant'
        # Example: .../features/all_features -> .../features/figures/significant
        output_dir = all_features_dir.parent / "figures" / "significant"
    else:
        output_dir = Path(output_dir)

    # Validate input directory
    if not all_features_dir.exists():
        msg = f"Directory does not exist: {all_features_dir}"
        raise FileNotFoundError(msg)
    if not all_features_dir.is_dir():
        msg = f"Path is not a directory: {all_features_dir}"
        raise NotADirectoryError(msg)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(all_features_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {all_features_dir}")
        return {"created": 0, "skipped": 0, "failed": 0}

    logger.info(f"Found {len(csv_files)} CSV files in {all_features_dir}")

    # Validate n_jobs
    if n_jobs <= 0:
        raise ValueError(f"n_jobs must be greater than 0. Your value: {n_jobs}")

    # Prepare job parameters
    jobs = [
        (
            csv_path,
            output_dir,
            overwrite,
            number_of_features,
            top_features,
            dpi,
            kwargs,
        )
        for csv_path in csv_files
    ]

    # Process files (sequential or parallel)
    job_results = None
    if n_jobs == 1:
        # Sequential processing
        job_results = [_process_single_file(*job) for job in jobs]  # type: ignore[arg-type]
    else:
        # Parallel processing
        logger.info(f"Running on {n_jobs} job(s)")
        with Pool(n_jobs) as pool:
            job_results = pool.starmap(_process_single_file, jobs)

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
