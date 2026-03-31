"""Eruption probability forecast visualisation with multi-classifier panels.

Renders the output of ``ModelPredictor.predict_proba()`` as time-series charts
showing per-classifier eruption probability, uncertainty bands, and consensus
statistics. Eruption event dates are marked with vertical lines when provided.

Key functions:

- ``plot_forecast(df, ...)`` — main entry point; creates one subplot panel per
  classifier present in the DataFrame plus a consensus panel at the bottom.
  Shades uncertainty envelopes and draws a horizontal probability threshold line.
  Accepts an optional list of eruption dates to annotate.
- ``plot_forecast_from_file(csv_path, ...)`` — convenience wrapper that loads a
  forecast CSV produced by ``ModelPredictor`` and delegates to ``plot_forecast``.

Internal helpers (not exported):

- ``_ax_per_classifier`` — renders a single classifier panel onto a given ``Axes``.
- ``_ax_forecast`` — renders the consensus probability panel.
- ``_ax_eruption`` — adds vertical eruption-event annotations to any ``Axes``.
"""

from typing import Any
from datetime import datetime

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from eruption_forecast.utils.ml import get_classifier_label
from eruption_forecast.plots.styles import DIVERGING_BREWER
from eruption_forecast.utils.dataframe import get_envelope_values
from eruption_forecast.utils.date_utils import (
    sort_dates,
    set_datetime_index,
)


def plot_forecast(
    df: pd.DataFrame,
    label_df: pd.DataFrame | pd.Series | None = None,
    title: str | None = None,
    fig_width: float = 12,
    fig_height: float = 3,
    threshold: float = 0.7,
    rolling_window: str = "6h",
    eruption_dates: list[str] | None = None,
) -> plt.Figure:
    """Plot eruption forecast probability and prediction time-series with Nature/Science styling.

    Creates a three-panel plot from a multi-model consensus DataFrame:
    - Panel 1 (top): Consensus max-envelope prediction and probability, with threshold line.
    - Panel 2 (middle): Per-classifier predictions overlaid with the consensus
      prediction envelope.
    - Panel 3 (bottom): Per-classifier probabilities overlaid with the consensus
      probability envelope, with formatted x-axis date labels.

    All panels use ``_ax_forecast`` fill-between coloring to highlight regions above the
    threshold (red), in the tolerance band (yellow), and below (green). Eruption
    dates are marked as vertical dashed lines on every panel when provided.

    Args:
        df (pd.DataFrame): Forecast consensus DataFrame with a datetime index and columns
            following the pattern ``{classifier_name}_prediction``,
            ``{classifier_name}_probability``, ``consensus_prediction_max_envelope``,
            ``consensus_prediction_min_envelope``, ``consensus_probability_max_envelope``,
            and ``consensus_probability_min_envelope``. Classifier names are inferred
            automatically from the column prefixes (everything before the first ``_``).
        label_df (pd.DataFrame | pd.Series): Label DataFrame or Series used to align ``df``
            to the label datetime index via :func:`set_datetime_index`.
        title (str | None, optional): Figure suptitle. Defaults to ``"Forecast Results"``
            when ``None``.
        fig_width (float, optional): Figure width in inches. Defaults to ``12``.
        fig_height (float, optional): Height of each individual panel in inches; total
            figure height is ``3 * fig_height``. Defaults to ``3``.
        threshold (float, optional): Decision threshold drawn as a horizontal dashed
            line on every panel and used for fill-between coloring. Defaults to ``0.7``.
        rolling_window (str, optional): Pandas-compatible window string passed to
            ``DataFrame.rolling()`` for smoothing before plotting. Defaults to ``"6h"``.
        eruption_dates (list[str] | None, optional): Eruption dates to annotate on
            every panel as vertical dashed lines. Each entry is passed to
            :func:`to_datetime`. Defaults to ``None``.

    Returns:
        plt.Figure: Matplotlib figure object with three vertically stacked subplots.

    Examples:
        >>> fig = plot_forecast(df, label_df, title="OJN 2025-03", eruption_dates=["2025-03-20"])
        >>> fig.savefig("forecast.png", dpi=150, bbox_inches="tight")
    """
    model_names: list[str] = list(
        {
            column.split("_")[0]
            for column in df.columns.tolist()
            if not column.startswith("consensus")
        }
    )

    # Ensure df have pd.DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if label_df is None or len(label_df) == 0:
            raise ValueError(
                "label_df is needed since concensus dataframe does not have pd.DatetimeIndex."
            )
        df = set_datetime_index(label_df, df)

    # Maintain backward compatibility
    # Old dataframe using "_eruption_probability" as suffix column name
    for column in df.columns:
        if column.endswith("eruption_probability"):
            df = df.rename(
                columns={column: column.replace("eruption_probability", "probability")}
            )

    df = get_envelope_values(df)

    # Smoothing
    df_resampled = df.rolling(window=rolling_window).mean()

    # Plot figure
    fig, axs = plt.subplots(
        nrows=3, ncols=1, figsize=(fig_width, 3 * fig_height), sharex=True
    )

    for index in range(3):
        ax = axs[index]

        # Consensus Prediction and Probability Plot
        if index == 0:
            ax.plot(
                df_resampled.index,
                df_resampled["consensus_prediction_max_envelope"],
                color="red",
                label="Cons. Prediction",
                linewidth=1.2,
                linestyle="-",
            )
            ax.plot(
                df_resampled.index,
                df_resampled["consensus_probability_max_envelope"],
                color="blue",
                label="Cons. Probability",
                linewidth=1.2,
                linestyle="-",
            )

            ax.plot(
                df.index,
                df["consensus_probability_max_envelope"],
                color="#0072B2",
                alpha=0.6,
                linewidth=0.5,
                linestyle="-",
            )

            ax = _ax_forecast(
                ax=ax,
                df_=df_resampled,
                max_column="consensus_prediction_max_envelope",
                threshold=threshold,
            )

            ax.set_ylabel("Consensus", fontsize=12)

        # Per Classifiers Consensus Prediction
        if index == 1:
            for _index, model_name in enumerate(model_names):
                ax = _ax_per_classifier(
                    df_resampled,
                    ax,
                    "prediction",
                    model_name,
                    color=DIVERGING_BREWER[_index],
                )

            ax = _ax_forecast(
                ax=ax,
                df_=df_resampled,
                max_column="consensus_prediction_max_envelope",
                min_column="consensus_prediction_min_envelope",
                threshold=threshold,
            )

            ax.set_ylabel("Cons. Prediction", fontsize=12)

        # Per Classifiers Consensus Probability
        if index == 2:
            for _index, model_name in enumerate(model_names):
                ax = _ax_per_classifier(
                    df_resampled,
                    ax,
                    "probability",
                    model_name,
                    color=DIVERGING_BREWER[_index],
                )

            ax = _ax_forecast(
                ax=ax,
                df_=df_resampled,
                max_column="consensus_probability_max_envelope",
                min_column="consensus_probability_min_envelope",
                threshold=threshold,
            )

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=30, horizontalalignment="right")

            ax.set_ylabel("Cons. Probability", fontsize=12)

        if eruption_dates is not None and len(eruption_dates) > 0:
            _eruption_dates = sort_dates(eruption_dates, as_datetime=True)

            for _index, eruption_date in enumerate(_eruption_dates):
                label = "Eruption" if _index == 0 else None
                if df.index[0] <= eruption_date <= df.index[-1]:
                    ax = _ax_eruption(ax, eruption_date, label=label)

        ax.axhline(
            y=threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold {threshold}",
        )

        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=12)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    # Collect unique handles/labels from all panels into one shared legend
    handles, labels = [], []
    seen = set()
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)

    fig.legend(
        handles,
        labels,
        loc="lower left",
        ncol=5,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.05, -0.075),
    )

    fig.suptitle(title or "Forecast Results", fontsize=14)
    plt.tight_layout()

    return fig


def plot_forecast_from_file(
    consensus_file: str,
    label_file: str | None = None,
    title: str | None = None,
    fig_width: float = 12,
    fig_height: float = 3,
    threshold: float = 0.7,
    rolling_window: str = "6h",
    eruption_dates: list[str] | None = None,
) -> plt.Figure:
    """Load consensus and label CSVs from disk and plot the forecast.

    Reads both files, aligns the consensus DataFrame to the label datetime index
    via :func:`set_datetime_index`, and delegates to :func:`plot_forecast`.

    Args:
        consensus_file (str): Path to the consensus forecast CSV with an ``id``
            index column and per-classifier/consensus output columns.
        label_file (str): Path to the label CSV used for datetime alignment.
        title (str | None, optional): Figure suptitle. Defaults to
            ``"Forecast Results"`` when ``None``.
        fig_width (float, optional): Figure width in inches. Defaults to ``12``.
        fig_height (float, optional): Height of each individual panel in inches;
            total figure height is ``3 * fig_height``. Defaults to ``3``.
        threshold (float, optional): Decision threshold for fill-between coloring.
            Defaults to ``0.7``.
        rolling_window (str, optional): Pandas-compatible window string for
            smoothing before plotting. Defaults to ``"6h"``.
        eruption_dates (list[str] | None, optional): Eruption dates to annotate
            as vertical dashed lines on every panel. Defaults to ``None``.

    Returns:
        plt.Figure: Matplotlib figure object with three vertically stacked subplots.
    """
    df = pd.read_csv(consensus_file, index_col=0, parse_dates=True)
    label_df = (
        pd.read_csv(label_file, index_col=0, parse_dates=True)
        if label_file
        else pd.DataFrame()
    )

    return plot_forecast(
        df,
        label_df,
        title,
        fig_width,
        fig_height,
        threshold,
        rolling_window,
        eruption_dates,
    )


def _ax_per_classifier(
    df_: pd.DataFrame,
    ax: plt.Axes,
    column_name: str,
    model_name: str,
    color: str = "#000000",
) -> plt.Axes:
    """Plot a single classifier's prediction or probability series on the given axes.

    Args:
        df_ (pd.DataFrame): Smoothed forecast DataFrame containing a column
            named ``{model_name}_{column_name}``.
        ax (plt.Axes): Matplotlib axes to plot on.
        column_name (str): Column suffix to plot; either ``"prediction"`` or
            ``"probability"``.
        model_name (str): Classifier name prefix used to resolve the column and
            the human-readable legend label via :func:`get_classifier_label`.
        color (str, optional): Line color. Defaults to ``"#000000"``.

    Returns:
        plt.Axes: The modified axes object.
    """
    label = get_classifier_label(model_name)
    column = f"{model_name}_{column_name}"
    ax.plot(
        df_.index, df_[column], color=color, linewidth=1.2, label=label, linestyle="-."
    )
    return ax


def _ax_forecast(
    ax: plt.Axes,
    df_: pd.DataFrame,
    max_column: str,
    threshold: float,
    threshold_tolerance: float = 0.1,
    min_column: str | None = None,
    zorder: int = 100,
) -> plt.Axes:
    """Fill regions between envelope boundaries relative to the decision threshold.

    Applies three ``fill_between`` layers using ``max_column`` as the upper boundary
    and either zero or ``min_column`` as the lower boundary:

    - Red (alpha 0.5): max envelope is at or above ``threshold``.
    - Yellow (alpha 0.5): max envelope is within ``threshold_tolerance`` below
      ``threshold``.
    - Green (alpha 0.2): max envelope is below the tolerance band.

    Args:
        ax (plt.Axes): Matplotlib axes to fill on.
        df_ (pd.DataFrame): Smoothed forecast DataFrame.
        max_column (str): Column name for the upper envelope boundary.
        threshold (float): Decision threshold for region classification.
        threshold_tolerance (float, optional): Width of the warning band below
            the threshold. Defaults to ``0.1``.
        min_column (str | None, optional): Column name for the lower envelope
            boundary. When ``None``, the lower boundary is fixed at zero.
            Defaults to ``None``.
        zorder (int, optional): Base z-order for fill layers; each subsequent
            layer is drawn one level lower. Defaults to ``100``.

    Returns:
        plt.Axes: The modified axes object.
    """

    min_value = 0 if min_column is None else df_[min_column]
    max_value = df_[max_column]

    labels: list[dict[str, Any]] = [
        {
            "name": f"p>={threshold}",
            "where": max_value >= threshold,
            "color": "#d73027",
            "alpha": 0.5,
        },
        {
            "name": f"(0.6<p<={threshold})",
            "where": (max_value >= (threshold - threshold_tolerance))
            & (max_value < threshold),
            "color": "#fee090",
            "alpha": 0.5,
        },
        {
            "name": f"p<={threshold - threshold_tolerance}",
            "where": max_value <= (threshold - threshold_tolerance),
            "color": "#009E73",
            "alpha": 0.2,
        },
    ]

    for index, label in enumerate(labels):
        ax.fill_between(
            df_.index,
            min_value,
            max_value,
            where=label["where"],
            color=label["color"],
            label=label["name"],
            zorder=zorder - index,
            alpha=label["alpha"],
        )

    return ax


def _ax_eruption(
    ax: plt.Axes, eruption_date: datetime, label: str | None = None
) -> plt.Axes:
    """Annotate a single eruption date on the given axes as a vertical dashed line.

    Draws a red dashed vertical line and a rotated date label at the eruption date.

    Args:
        ax (plt.Axes): Matplotlib axes to annotate.
        eruption_date (datetime): Eruption date to mark.
        label (str | None, optional): Legend label for the line. Pass ``None`` to
            suppress the legend entry for subsequent eruptions. Defaults to ``None``.

    Returns:
        plt.Axes: The modified axes object.
    """
    ax.axvline(
        x=eruption_date,  # ty:ignore[invalid-argument-type]
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=label,
    )

    ax.text(
        x=eruption_date,  # ty:ignore[invalid-argument-type]
        y=0.02,
        s=eruption_date.strftime("%Y-%m-%d"),
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=100,
    )

    return ax
