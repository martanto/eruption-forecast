"""Feature importance and selection visualization with Nature/Science styling."""

import os

import pandas as pd
import matplotlib.pyplot as plt

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
        plt.rcParams['figure.constrained_layout.use'] = False

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # Color bars by position: top features in darker blue
        bar_colors = [
            NATURE_COLORS["blue"] if i >= (number_of_features - top_features)
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
            ax.legend(frameon=False, loc="lower right")

        # Configure axes
        configure_spine(ax)
        ax.set_xlabel(
            "P-value" if values_column == "p_values" else "Importance Score"
        )
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

        # Apply tight layout to prevent label clipping
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi)
        plt.close()

    return None
