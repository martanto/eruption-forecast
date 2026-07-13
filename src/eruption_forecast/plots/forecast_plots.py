from typing import Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import DIVERGING_BREWER
from eruption_forecast.utils.dataframe import get_envelope_values
from eruption_forecast.utils.date_utils import (
    sort_dates,
    to_datetime,
    to_datetime_index,
)
from eruption_forecast.utils.formatting import get_classifier_label


def plot_forecast(
    df: pd.DataFrame,
    label_df: pd.DataFrame | pd.Series | None = None,
    title: str | None = None,
    fig_width: float = 8,
    fig_height: float = 1.5,
    threshold: float = 0.7,
    rolling_window: str = "6h",
    x_days_interval: int = 2,
    x_label_fontsize: int = 7,
    x_label_rotation: int = -15,
    eruption_dates: list[str] | None = None,
    y_max: float = 1.05,
    legend_n_cols: int = 6,
    legend_fontsize: int = 6,
    legend_position: tuple[float, float] = (0.5, -0.075),
    training_start_date: str | None = None,
    training_end_date: str | None = None,
    prediction_start_date: str | None = None,
    prediction_end_date: str | None = None,
) -> plt.Figure:
    """Plot eruption forecast probability and prediction time-series.

    Creates a stacked plot from a multi-model consensus DataFrame. The base
    layout has three panels:

    - Panel 1 (top): Consensus max-envelope prediction and probability, with
      threshold line.
    - Panel 2 (middle): Per-classifier predictions overlaid with the consensus
      prediction envelope.
    - Panel 3 (bottom): Per-classifier probabilities overlaid with the consensus
      probability envelope, with formatted x-axis date labels.

    When all four ``training_start_date``, ``training_end_date``,
    ``prediction_start_date``, and ``prediction_end_date`` are supplied, a
    fourth thin strip is added at the top showing training window → gap →
    prediction window as a Gantt-style horizontal bar. The strip shares the
    x-axis with the three data panels.

    All data panels use ``_ax_forecast`` fill-between coloring to highlight
    regions above the threshold (red), in the tolerance band (yellow), and
    below (green). Eruption dates are marked as vertical dashed lines on
    every data panel when provided.

    Args:
        df (pd.DataFrame): Forecast consensus DataFrame with a datetime index and columns
            following the pattern ``{classifier_name}_prediction``,
            ``{classifier_name}_probability``, ``consensus_prediction_max_envelope``,
            ``consensus_prediction_min_envelope``, ``consensus_probability_max_envelope``,
            and ``consensus_probability_min_envelope``. Classifier names are inferred
            automatically from the column prefixes (everything before the first ``_``).
        label_df (pd.DataFrame | pd.Series | None, optional): Label DataFrame or Series
            used to align ``df`` to the label datetime index via
            :func:`to_datetime_index`. Required when ``df.index`` is not already a
            ``pd.DatetimeIndex``. Defaults to ``None``.
        title (str | None, optional): Figure suptitle. Defaults to ``"Forecast Results"``
            when ``None``.
        fig_width (float, optional): Figure width in inches. Defaults to ``8``.
        fig_height (float, optional): Height of each individual data panel in inches;
            total figure height is ``3 * fig_height`` (plus a small increment for the
            top segment strip when it is rendered). Defaults to ``1.5``.
        threshold (float, optional): Decision threshold drawn as a horizontal dashed
            line on every data panel and used for fill-between coloring. Defaults to
            ``0.7``.
        rolling_window (str, optional): Pandas-compatible window string passed to
            ``DataFrame.rolling()`` for smoothing before plotting. Defaults to ``"6h"``.
        x_days_interval (int, optional): X-axis days interval. Defaults to ``2``.
        x_label_fontsize (int, optional): X-axis tick-label font size on the bottom
            data panel. Defaults to ``7``.
        x_label_rotation (int, optional): X-axis tick-label rotation in degrees on the
            bottom data panel. Defaults to ``-15``.
        eruption_dates (list[str] | None, optional): Eruption dates to annotate on
            every data panel as vertical dashed lines. Each entry is passed to
            :func:`to_datetime`. Defaults to ``None``.
        y_max (float, optional): Max y-value for the data panels. Defaults to ``1.05``.
        legend_n_cols (int, optional): Number of columns for the shared bottom legend.
            Defaults to ``6``.
        legend_fontsize (int, optional): Font size for the shared bottom legend.
            Defaults to ``6``.
        legend_position (tuple[float, float], optional): Position of the shared bottom
            legend, forwarded as ``bbox_to_anchor`` to ``fig.legend()``.
            Defaults to ``(0.5, -0.075)``.
        training_start_date (str | None, optional): Start of the training window for
            the top segment strip (green bar). The strip is only rendered when all
            four ``training_*`` / ``prediction_*`` dates are non-``None``.
            Defaults to ``None``.
        training_end_date (str | None, optional): End of the training window for the
            top segment strip. Defaults to ``None``.
        prediction_start_date (str | None, optional): Start of the prediction window
            for the top segment strip (red bar). Defaults to ``None``.
        prediction_end_date (str | None, optional): End of the prediction window for
            the top segment strip. Defaults to ``None``.

    Returns:
        plt.Figure: Matplotlib figure with three vertically stacked data panels, plus
            an additional top segment strip when all four ``training_*`` /
            ``prediction_*`` dates are supplied.

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
                "df must have a DatetimeIndex; provide label_df (id→datetime mapping) so "
                "to_datetime_index() can construct and align the forecast index."
            )
        df = to_datetime_index(label_df, df)

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
    show_segment = all(
        d is not None
        for d in (
            training_start_date,
            training_end_date,
            prediction_start_date,
            prediction_end_date,
        )
    )

    if show_segment:
        fig, all_axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(fig_width, 3 * fig_height + 0.4),
            gridspec_kw={"height_ratios": [0.2, 1, 1, 1], "hspace": 0.05},
            sharex=True,
            constrained_layout=True,
        )
        ax_segment, axs = all_axs[0], all_axs[1:]
    else:
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(fig_width, 3 * fig_height),
            sharex=True,
            tight_layout=True,
        )
        ax_segment = None

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

            ax.set_ylabel("Consensus", fontsize=8)

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

            ax.set_ylabel("Cons. Prediction", fontsize=8)

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

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_days_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            tick_labels = ax.get_xticklabels(which="major")
            for label in tick_labels:
                label.set(
                    rotation=x_label_rotation,
                    horizontalalignment="left",
                    fontsize=x_label_fontsize,
                )

            ax.set_xlim(df.index.min(), df.index.max())
            ax.set_ylabel("Cons. Probability", fontsize=8)

        if eruption_dates is not None and len(eruption_dates) > 0:
            _eruption_dates = sort_dates(eruption_dates, as_datetime=True)

            for _index, eruption_date in enumerate(_eruption_dates):
                label = "Eruption" if _index == (len(_eruption_dates) - 1) else None
                if df.index[0] <= eruption_date <= df.index[-1]:
                    ax = ax_eruption(ax, to_datetime(eruption_date), label=label)

        ax.axhline(
            y=threshold,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label=f"Threshold {threshold}",
        )

        ax.set_ylim(0, y_max)
        ax.tick_params(labelsize=6)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    if (
        ax_segment is not None
        and training_start_date is not None
        and training_end_date is not None
        and prediction_start_date is not None
        and prediction_end_date is not None
    ):
        _ax_segment(
            ax_segment,
            training_start_date,
            training_end_date,
            prediction_start_date,
            prediction_end_date,
        )

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
        loc="lower center",
        ncol=legend_n_cols,
        fontsize=legend_fontsize,
        frameon=False,
        fancybox=False,
        bbox_to_anchor=legend_position,
    )

    start_date_str = df.index.min().strftime("%Y-%m-%d")
    end_date_str = df.index.max().strftime("%Y-%m-%d")
    default_title = f"Forecast Results\n{start_date_str}-{end_date_str}"

    fig.suptitle(title or default_title, fontsize=8)

    return fig


def plot_forecast_from_file(
    consensus_file: str,
    label_file: str | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Load consensus and (optionally) label CSVs from disk and plot the forecast.

    Reads ``consensus_file``, aligns the resulting DataFrame to the label datetime
    index via :func:`to_datetime_index` when ``label_file`` is provided, and
    delegates to :func:`plot_forecast`. All extra keyword arguments are forwarded
    verbatim to :func:`plot_forecast` — see that function for the full list of
    styling / layout options and the optional top segment strip.

    Args:
        consensus_file (str): Path to the consensus forecast CSV with an ``id``
            index column and per-classifier/consensus output columns.
        label_file (str | None, optional): Path to the label CSV used for datetime
            alignment. When ``None``, an empty label frame is passed through and
            :func:`plot_forecast` will raise unless ``consensus_file`` already
            carries a ``pd.DatetimeIndex``. Defaults to ``None``.
        **kwargs: Forwarded to :func:`plot_forecast` (e.g. ``title``, ``fig_width``,
            ``fig_height``, ``threshold``, ``rolling_window``, ``x_days_interval``,
            ``x_label_fontsize``, ``x_label_rotation``, ``eruption_dates``, ``y_max``,
            ``legend_n_cols``, ``legend_fontsize``, ``legend_position``,
            ``training_start_date``, ``training_end_date``, ``prediction_start_date``,
            ``prediction_end_date``).

    Returns:
        plt.Figure: Matplotlib figure with three vertically stacked data panels, plus
            an additional top segment strip when all four ``training_*`` /
            ``prediction_*`` dates are supplied.
    """
    df = pd.read_csv(consensus_file, index_col=0, parse_dates=True)
    label_df = (
        pd.read_csv(label_file, index_col=0, parse_dates=True)
        if label_file
        else pd.DataFrame()
    )

    return plot_forecast(df=df, label_df=label_df, **kwargs)


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
            "name": f"p>={threshold}",  # eruption
            "where": max_value >= threshold,
            "color": "#d73027",
            "alpha": 0.5,
        },
        {
            "name": f"0.6<p<{threshold}",  # pre-eruption
            "where": (max_value >= (threshold - threshold_tolerance))
            & (max_value < threshold),
            "color": "#fee090",
            "alpha": 0.5,
        },
        {
            "name": f"p<={threshold - threshold_tolerance}",  # no eruption
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


def ax_eruption(
    ax: plt.Axes,
    eruption_date: datetime,
    label: str | None = None,
    fill_between: bool = True,
    fill_between_y_max: float = 1.05,
) -> plt.Axes:
    """Annotate a single eruption date on the given axes as a vertical dashed line.

    Draws a red dashed vertical line and a rotated date label at the eruption date.

    Args:
        ax (plt.Axes): Matplotlib axes to annotate.
        eruption_date (datetime): Eruption date to mark.
        label (str | None, optional): Legend label for the line. Pass ``None`` to
            suppress the legend entry for subsequent eruptions. Defaults to ``None``.
        fill_between_y_max (float, optional): Max y-value for the label. Defaults to ``1.05``.

    Returns:
        plt.Axes: The modified axes object.
    """
    start_eruption = eruption_date.replace(hour=0, minute=0, second=0)
    end_eruption = eruption_date.replace(hour=23, minute=59, second=0)

    ax.axvline(
        x=eruption_date,  # ty:ignore[invalid-argument-type]
        color="red",
        linewidth=1.0,
        linestyle="--",
        label=label,
    )

    if fill_between:
        ax.fill_between(
            np.array([start_eruption, end_eruption]),
            0.0,
            fill_between_y_max,
            color="#a50026",
            alpha=0.1,
        )

    ax.text(
        x=eruption_date,  # ty:ignore[invalid-argument-type]
        y=0.02,
        s=eruption_date.strftime("%Y-%m-%d"),
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="black",
        zorder=100,
    )

    return ax


def _ax_segment(
    ax: plt.Axes,
    training_start_date: str | datetime,
    training_end_date: str | datetime,
    prediction_start_date: str | datetime,
    prediction_end_date: str | datetime,
) -> plt.Axes:
    training_start = to_datetime(training_start_date).replace(
        hour=0, minute=0, second=0
    )
    training_end = to_datetime(training_end_date).replace(hour=23, minute=59, second=59)
    prediction_start = to_datetime(prediction_start_date).replace(
        hour=0, minute=0, second=0
    )
    prediction_end = to_datetime(prediction_end_date).replace(
        hour=23, minute=59, second=59
    )

    gap_duration = (prediction_start - training_end).days

    if gap_duration < 0:
        original_prediction_start = prediction_start
        prediction_start = (training_end + timedelta(days=1)).replace(
            hour=0, minute=0, second=0
        )
        logger.warning(
            f"Overlapping training end date ({training_end:%Y-%m-%d}) and "
            f"prediction start date ({original_prediction_start:%Y-%m-%d}). "
            f"Adjusted prediction start to {prediction_start:%Y-%m-%d}."
        )
        gap_duration = (prediction_start - training_end).days

    training_duration = (training_end - training_start).days + 1
    prediction_duration = (prediction_end - prediction_start).days + 1

    segments: list[dict[str, Any]] = [
        {
            "label": "Training",
            "width": training_duration,
            "left": mdates.date2num(training_start),
            "color": "#009E73",
            "hatch": None,
        },
        {
            "label": "Gap",
            "width": gap_duration,
            "left": mdates.date2num(training_end),
            "color": "white",
            "hatch": "////",
        },
        {
            "label": "Prediction",
            "width": prediction_duration,
            "left": mdates.date2num(prediction_start),
            "color": "#d73027",
            "hatch": None,
        },
    ]

    y_pos = -0.25
    for segment in segments:
        if segment["label"] == "Gap" and gap_duration == 0:
            continue

        ax.barh(
            y=y_pos,
            label=segment["label"],
            width=segment["width"],
            left=segment["left"],
            color=segment["color"],
            hatch=segment["hatch"],
            height=0.2,
            alpha=0.5,
        )

        if segment["label"] != "Gap":
            ax.text(
                segment["left"] + segment["width"] / 2,
                y_pos + 0.15,
                segment["label"],
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
    )

    ax.set_ylim(-0.2, 0.2)

    return ax
