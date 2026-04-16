"""Feature importance and selection visualisation.

Provides plotting functions for inspecting the tsfresh features selected during
training, including bar charts of top features ranked by importance score and
frequency-band contribution summaries. Supports both single-seed and batch
(multi-seed) workflows.

Key functions:

- ``plot_significant_features(features_df, ...)`` — horizontal bar chart of the
  top-N significant features for a single seed, coloured by tremor column origin
  (RSAM, DSAR, entropy).
- ``replot_significant_features(features_dir, ...)`` — batch replot all feature
  importance CSVs in a directory, with optional parallel processing via ``n_jobs``.
- ``plot_frequency_band_contribution(features_df, ...)`` — stacked bar chart showing
  what proportion of selected features comes from each frequency band, broken down
  by metric type (RSAM vs. DSAR vs. entropy).
"""

import os
import re
from typing import Any, Literal, cast
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    configure_spine,
    apply_nature_style,
)


def plot_significant_features(
    df: pd.DataFrame,
    filepath: str,
    number_of_features: int = 30,
    top_features: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] = (4, 12),
    features_column: str = "features",
    values_column: str = "score",
    dpi: int = 150,
    overwrite: bool = True,
) -> None:
    """Plot a horizontal bar chart of significant features with publication-quality styling.

    Displays the top ``number_of_features`` rows of ``df`` as a horizontal
    bar chart sorted by ``values_column``, with a dashed red reference line at
    ``top_features``. Top features are highlighted with blue, others with gray.
    Uses Nature/Science journal styling with colorblind-safe colors.

    Args:
        df (pd.DataFrame): DataFrame containing feature names and their
            significance values (e.g., p-values or importance scores). Must contain
            columns specified by features_column and values_column.
        filepath (str): Full path (including filename and extension) where the
            figure is saved. Parent directories must exist.
        number_of_features (int, optional): Total number of features to
            display in the chart. Features are sorted by values_column.
            Defaults to 50.
        top_features (int, optional): Position at which to draw a reference
            line marking the top-N cut-off. Top features are colored blue, others
            gray. Defaults to 20.
        title (str | None, optional): Chart title. If None, defaults to
            "<number_of_features> Significant Features". Defaults to None.
        figsize (tuple[float, float], optional): Figure dimensions as
            (width, height) in inches. If using the default value of (4, 12),
            the height will be automatically calculated based on number_of_features
            to prevent layout collapse (formula: max(8, number_of_features * 0.3 + 2)).
            Custom values are respected as-is. Defaults to (4, 12).
        features_column (str, optional): Name of the column containing
            feature names. If missing, the DataFrame index is used. Defaults to
            "features".
        values_column (str, optional): Name of the column containing
            significance values (e.g., p-values or importance scores). Must be
            numeric. Defaults to "score".
        dpi (int, optional): Figure resolution in dots per inch for saved PNG.
            Defaults to 150.
        overwrite (bool, optional): If True, overwrite an existing file at
            filepath. If False, skip plotting if file exists. Defaults to True.

    Returns:
        None: Saves figure to disk, does not return matplotlib objects.

    Examples:
        >>> # Basic usage with default parameters
        >>> plot_significant_features(
        ...     df=sig_features_df,
        ...     filepath="output/figures/significant_features.png",
        ...     number_of_features=30,
        ...     top_features=10,
        ... )
        >>>
        >>> # Custom column names for importance scores
        >>> plot_significant_features(
        ...     df=importance_df,
        ...     filepath="output/importance.png",
        ...     features_column="feature_name",
        ...     values_column="importance_score",
        ...     title="Top Feature Importances",
        ... )
    """
    number_of_features = (
        number_of_features if len(df.index) >= number_of_features else top_features
    )

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
        ax.set_xlabel("Score")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"{number_of_features} Significant Features")

        # Set y-axis limits
        ax.set_ylim(-0.5, number_of_features - 0.5)

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
    """Process a single CSV file and generate a feature importance plot.

    Helper function for multiprocessing support in replot_significant_features().
    Loads a CSV, auto-detects feature and value columns, and generates a plot
    using plot_significant_features(). Designed for use with Pool.starmap().

    Args:
        csv_path (Path): Path to input CSV file containing feature data.
        output_dir (Path): Directory where output plot will be saved.
        overwrite (bool): If True, overwrite existing plots. If False, skip
            plotting if file already exists.
        number_of_features (int): Number of features to display in the plot.
        top_features (int): Number of top features to highlight with blue color.
        dpi (int): Plot resolution in dots per inch.
        kwargs (dict): Additional keyword arguments for plot_significant_features(),
            such as "features_column", "values_column", "title", "figsize".

    Returns:
        str: Processing status - one of "created", "skipped", or "failed".

    Raises:
        Does not raise exceptions. Errors are logged via logger.error() and
        "failed" status is returned.

    Notes:
        - Auto-detects features column (tries "features" or uses index)
        - Auto-detects values column with priority: "score" -> "p_values" -> "importance" -> first numeric column
        - Output filename matches input CSV filename with .png extension
        - Errors are logged but don't raise exceptions for robustness
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
            # Prefer well-known legacy names before falling back to first numeric column
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                msg = f"No numeric columns found in {csv_path.name}"
                raise ValueError(msg)

            # Priority order for backward compatibility
            preferred_order = ["score", "p_values", "importance"]
            for col_name in preferred_order:
                if col_name in numeric_cols:
                    values_column = col_name
                    break

            # Fallback to first numeric column if no preferred column is found
            if values_column is None:
                values_column = numeric_cols[0]

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
            - "created": Number of plots successfully created
            - "skipped": Number of plots skipped (file exists, overwrite=False)
            - "failed": Number of plots that failed due to errors

    Raises:
        FileNotFoundError: If all_features_dir does not exist.
        NotADirectoryError: If all_features_dir is not a directory.
        ValueError: If n_jobs is less than or equal to 0.

    Examples:
        >>> # Replot all features with default settings
        >>> results = replot_significant_features(
        ...     all_features_dir="output/.../features/all_features",
        ...     overwrite=True,
        ... )
        >>> print(f"Created: {results['created']}, Failed: {results['failed']}")

        >>> # Custom output directory with top 30 features
        >>> results = replot_significant_features(
        ...     all_features_dir="path/to/features",
        ...     output_dir="path/to/plots",
        ...     overwrite=False,
        ...     number_of_features=30,
        ...     top_features=15,
        ...     dpi=300,
        ... )

    Notes:
        - CSV files are expected to have either a 'features' column or feature
          names in the index.
        - The function attempts to auto-detect the values column (uses first
          numeric column, expected to be "score").
        - Errors are logged but don't stop processing of remaining files.
        - Output filenames match input CSV filenames with .png extension.
        - Default output directory is ``<parent>/figures/significant`` where
          ``<parent>`` is the parent directory of ``all_features_dir``.
    """
    # Convert paths to Path objects
    all_features_dir: Path = Path(all_features_dir)
    if output_dir is None:
        # Default: create sibling directory called 'figures/significant'
        # Example: .../features/all_features -> .../features/figures/significant
        output_dir: Path = all_features_dir.parent / "figures" / "significant"
    else:
        output_dir: Path = Path(output_dir)

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


# ---------------------------------------------------------------------------
# Frequency band contribution plot
# ---------------------------------------------------------------------------

_BAND_PREFIX_RE = re.compile(r"^((?:rsam|dsar)_[^_]+|entropy)")

# Color map: each calculate method → Okabe-Ito color (matches tremor_plots.py)
_METHOD_COLORS: dict[str, str] = {
    "rsam": OKABE_ITO[4],  # Blue
    "dsar": OKABE_ITO[0],  # Orange
    "entropy": OKABE_ITO[6],  # Reddish purple
}


def _extract_band_prefix(feature_name: str) -> str:
    """Extract the seismic band prefix from a tsfresh feature name.

    Parses the leading ``rsam_fN`` or ``dsar_fN-fM`` segment from a
    tsfresh feature string such as ``rsam_f2__mean`` or
    ``dsar_f0-f1__quantile__q_0.9``.

    Args:
        feature_name (str): Full tsfresh feature name.

    Returns:
        str: Band prefix (e.g. ``"rsam_f2"`` or ``"dsar_f0-f1"``), or the
            full string if no prefix is recognised.
    """
    match = _BAND_PREFIX_RE.match(feature_name)
    return match.group(1) if match else feature_name


def _annotate_bar_percentages(
    ax: plt.Axes,
    counts: pd.Series | list,
    total: float,
) -> None:
    """Annotate each horizontal bar with its percentage of the total count.

    Writes ``"X.X%"`` just to the right of each bar end, vertically centered
    on the bar.

    Args:
        ax (plt.Axes): Axes containing the horizontal bar chart.
        counts (pd.Series | list): Count values in the same order as the bars
            (bottom to top).
        total (float): Total count used as the denominator for percentages.
    """
    xmax = float(ax.get_xlim()[1])
    pad = xmax * 0.02
    x_right_limit = xmax * 0.98

    for i, count in enumerate(counts):
        pct = count / total * 100 if total > 0 else 0.0
        x_text = float(count) + pad
        ha = "left"
        if x_text > x_right_limit:
            x_text = x_right_limit
            ha = "right"
        ax.text(
            x_text,
            i,
            f" {int(count)} ({pct:.1f}%)",
            va="center",
            ha=ha,
            fontsize=8,
            clip_on=True,
        )


def _finalize_frequency_band_layout(fig: plt.Figure, ax: plt.Axes) -> None:
    """Adjust subplot margins to keep all labels visible in saved figures."""
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    tight_bbox = ax.get_tightbbox(renderer)
    if tight_bbox is None:
        fig.subplots_adjust(left=0.22, right=0.95)
        return
    bbox = tight_bbox.transformed(fig.transFigure.inverted())

    left = fig.subplotpars.left
    right = fig.subplotpars.right
    min_outer_margin = 0.01
    target_left_edge = 0.04
    target_right_edge = 1.0 - min_outer_margin

    if bbox.x0 < target_left_edge:
        left += target_left_edge - bbox.x0
    if bbox.x1 > target_right_edge:
        right -= bbox.x1 - target_right_edge

    left = min(max(left, 0.22), 0.5)
    right = max(min(right, 0.98), left + 0.3)
    fig.subplots_adjust(left=left, right=right)


def plot_frequency_band_contribution(
    feature_names: list[str] | list[list[str]],
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot the number of selected features grouped by seismic frequency band.

    Parses tsfresh feature names to extract band prefixes (e.g. ``rsam_f2``,
    ``dsar_f0-f1``) and counts how many features belong to each band. When
    ``feature_names`` is a list of lists (multi-seed), computes mean ± std
    count per band across seeds.

    RSAM bands are coloured blue, DSAR bands orange, and entropy bands
    reddish-purple to make the contribution of each measurement type
    immediately apparent.

    Args:
        feature_names (list[str] | list[list[str]]): Either a flat list of
            feature names (single seed) or a list of lists (one list per
            seed for multi-seed aggregation).
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (8, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. If None, uses
            "Feature Count by Frequency Band". Defaults to None.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``band`` and ``count``. For multi-seed input,
            ``count`` is the total across all seeds.

    Examples:
        >>> # Single seed
        >>> fig, df = plot_frequency_band_contribution(selected_feature_names)
        >>> fig.savefig("band_contribution.png")

        >>> # Multi-seed (one list of features per seed)
        >>> per_seed_features = [seed0_features, seed1_features, ...]
        >>> fig, df = plot_frequency_band_contribution(per_seed_features)
        >>> fig.savefig("band_contribution_aggregate.png")
    """
    is_multi_seed = feature_names and isinstance(feature_names[0], list)

    if is_multi_seed:
        # Count unique feature names across all seeds for the title
        unique_feature_count = len(
            {f for seed_feats in feature_names for f in seed_feats}
        )

        # Build count DataFrame per seed then aggregate
        seed_counts: list[pd.Series] = []
        for seed_features in feature_names:
            prefixes = [_extract_band_prefix(f) for f in seed_features]
            counts = pd.Series(prefixes).value_counts()
            seed_counts.append(counts)

        count_df = pd.DataFrame(seed_counts).fillna(0)
        total_counts = count_df.sum()

        # Sort by total count descending
        order = total_counts.sort_values(ascending=False).index.tolist()
        display_counts = total_counts[order].iloc[::-1]
        display_bands = order[::-1]

        bar_colors = [
            next(
                (
                    color
                    for method, color in _METHOD_COLORS.items()
                    if b.startswith(method)
                ),
                OKABE_ITO[-1],  # Fallback for unknown prefix
            )
            for b in display_bands
        ]

        with apply_nature_style():
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=False)
            ax.barh(
                range(len(display_bands)),
                display_counts,
                color=bar_colors,
                alpha=0.8,
            )
            max_count = float(display_counts.max()) if len(display_counts) > 0 else 0.0
            ax.set_xlim(0.0, max(max_count * 1.35, 1.0))
            ax.set_yticks(range(len(display_bands)))
            ax.set_yticklabels(display_bands)
            configure_spine(ax)
            ax.set_xlabel("Feature Count")
            ax.set_ylabel("Frequency Band", labelpad=10)
            ax.set_title(title or f"Feature Count ({unique_feature_count} unique)")
            legend_handles = [
                Patch(facecolor=color, alpha=0.8, label=method.upper())
                for method, color in _METHOD_COLORS.items()
            ]
            ax.legend(handles=legend_handles, frameon=False)
            _annotate_bar_percentages(ax, display_counts, total_counts.sum())
            _finalize_frequency_band_layout(fig, ax)

        data = pd.DataFrame(
            {
                "band": order,
                "count": total_counts[order].to_numpy(),
            }
        )

    else:
        # Single seed
        single_seed_features = cast(list[str], feature_names)
        prefixes = [_extract_band_prefix(f) for f in single_seed_features]
        counts = pd.Series(prefixes).value_counts()

        order = counts.index.tolist()
        display_bands = order[::-1]
        display_counts = counts.iloc[::-1]

        bar_colors = [
            next(
                (
                    color
                    for method, color in _METHOD_COLORS.items()
                    if b.startswith(method)
                ),
                OKABE_ITO[-1],  # Fallback for unknown prefix
            )
            for b in display_bands
        ]

        with apply_nature_style():
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=False)
            ax.barh(
                range(len(display_bands)),
                display_counts,
                color=bar_colors,
                alpha=0.8,
            )
            max_count = float(display_counts.max()) if len(display_counts) > 0 else 0.0
            ax.set_xlim(0.0, max(max_count * 1.35, 1.0))
            ax.set_yticks(range(len(display_bands)))
            ax.set_yticklabels(display_bands)
            configure_spine(ax)
            ax.set_xlabel("Feature Count")
            ax.set_ylabel("Frequency Band", labelpad=10)
            total = int(display_counts.sum())
            ax.set_title(title or f"Feature Count ({total})")
            legend_handles = [
                Patch(facecolor=color, alpha=0.8, label=method.upper())
                for method, color in _METHOD_COLORS.items()
            ]
            ax.legend(handles=legend_handles, frameon=False)
            _annotate_bar_percentages(ax, display_counts, counts.sum())
            _finalize_frequency_band_layout(fig, ax)

        data = pd.DataFrame({"band": order, "count": counts[order].to_numpy()})

    return fig, data


def plot_feature_correlations(
    resampled_df: pd.DataFrame,
    significant_features_df: pd.DataFrame,
    filepath: str,
    *,
    method: Literal["pearson", "spearman"] = "spearman",
    features_column: str = "features",
    top_features: int = 20,
    figsize: tuple[float, float] = (6, 6),
    cmap: str = "RdYlBu_r",
    dpi: int = 150,
    title: str | None = None,
    overwrite: bool = True,
) -> None:
    """Plot a correlation heatmap for the top-N significant features of a single seed.
    Subsets the resampled training matrix to the top-``top_features`` feature names
    listed in ``significant_features_df``, computes a pairwise correlation matrix using
    the requested ``method``, and renders it as a colour-mapped heatmap with annotated
    cell values.
    Args:
        resampled_df (pd.DataFrame): Resampled training feature matrix. Must contain
            the feature columns named in ``significant_features_df``. Any non-feature
            columns (e.g. ``id``, ``is_erupted``) are ignored automatically.
        significant_features_df (pd.DataFrame): DataFrame of selected features. Must
            contain a column named ``features_column`` with tsfresh feature names sorted
            by ascending score (most significant first).
        filepath (str): Full path (including filename and extension) where the figure
            is saved. Parent directory must exist.
        method (Literal["pearson", "spearman"], optional): Correlation method passed
            directly to ``pd.DataFrame.corr()``. Defaults to ``"spearman"``.
        features_column (str, optional): Name of the column in ``significant_features_df``
            that contains feature names. Defaults to ``"features"``.
        top_features (int, optional): Number of top features to include in the heatmap.
            Defaults to 20.
        figsize (tuple[float, float], optional): Figure dimensions in inches.
            Defaults to (6, 6).
        cmap (str, optional): Matplotlib colormap name for the heatmap. Defaults to
            ``"RdYlBu_r"`` (diverging, colourblind-friendly).
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. If None, defaults to
            ``"Feature Correlations (spearman)"`` (or pearson). Defaults to None.
        overwrite (bool, optional): If False, skip saving when ``filepath`` already
            exists. Defaults to True.
    Returns:
        None
    Examples:
        >>> sig_df = pd.read_csv("significant_features/00000.csv")
        >>> res_df = pd.read_csv("resampled/00000.csv", index_col=0)
        >>> plot_feature_correlations(res_df, sig_df, "figures/correlations/00000.png")
    """
    if not overwrite and os.path.isfile(filepath):
        logger.info(f"Correlation features plot {filepath} already exists.")
        return

    # Resolve feature names from significant_features_df.
    if features_column in significant_features_df.columns:
        feature_names = significant_features_df[features_column].tolist()
    else:
        # Fall back to the DataFrame index (e.g. when loaded with index_col=0).
        feature_names = significant_features_df.index.tolist()

    feature_names = feature_names[:top_features]

    # Keep only columns that actually exist in resampled_df.
    available = [f for f in feature_names if f in resampled_df.columns]
    missing = len(feature_names) - len(available)
    if missing:
        logger.warning(
            f"plot_feature_correlations: {missing} feature(s) not found in "
            "resampled_df and will be skipped."
        )
    if not available:
        logger.warning("plot_feature_correlations: no features to plot; skipping.")
        return

    corr: pd.DataFrame = resampled_df[available].corr(method=method)

    # Shorten tsfresh names for tick labels — keep last ~35 characters.
    _max_label_len = 35
    labels = [
        f"…{n[-_max_label_len:]}" if len(n) > _max_label_len + 1 else n
        for n in available
    ]

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, label=f"{method} correlation")

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels(labels, fontsize=5)
        configure_spine(ax)
        ax.set_title(title or f"Feature Correlations ({method})")

        # Annotate cells with 2-decimal values when the matrix is small enough.
        if len(available) <= 20:
            for i in range(len(available)):
                for j in range(len(available)):
                    val = corr.values[i, j]
                    # Choose white or black text based on absolute correlation strength.
                    text_color = "white" if abs(val) > 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4,
                        color=text_color,
                    )

        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
