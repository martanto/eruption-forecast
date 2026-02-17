"""Eruption forecast probability visualization with Nature/Science styling."""

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    configure_spine,
    apply_nature_style,
)


def plot_forecast(
    df: pd.DataFrame,
    model_names: list[str],
    multi_model: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 150,
    format_dates: bool = True,
) -> plt.Figure:
    """Plot eruption forecast probability and confidence time-series with Nature/Science styling.

    Creates a dual-panel plot showing eruption probability (top) and model confidence
    (bottom) over time. Supports both single-model and multi-model consensus modes.
    Multi-model mode shows individual classifier predictions plus consensus with uncertainty.

    Args:
        df (pd.DataFrame): Forecast DataFrame with datetime index and columns:
            - Single model: "eruption_probability", "confidence", "prediction"
              (or "{model_name}_eruption_probability", etc.)
            - Multi-model: "{name}_eruption_probability", "{name}_confidence" for
              each classifier, plus "consensus_eruption_probability",
              "consensus_uncertainty", "consensus_confidence"
        model_names (list[str]): List of model names (keys for multi-model columns).
            For single model, pass a list with one name, e.g., ["xgb"].
        multi_model (bool, optional): Whether this is a multi-model consensus plot.
            If True, plots individual classifiers with dashed lines and consensus
            with solid black line. If False, plots single model probability.
            Defaults to False.
        title (str | None, optional): Plot title suffix appended to "Eruption Forecast".
            If None, no suffix is added. Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (12, 6).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.
        format_dates (bool, optional): If True and index is datetime, format
            x-axis as dates with rotation. If False, use sequential window index.
            Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure object with two vertically stacked subplots.

    Examples:
        >>> # Single model forecast
        >>> fig = plot_forecast(df, model_names=["xgb"], multi_model=False)
        >>> fig.savefig("forecast_xgb.png")
        >>>
        >>> # Multi-model consensus
        >>> fig = plot_forecast(
        ...     df, model_names=["rf", "xgb", "gb"], multi_model=True,
        ...     title="3-Classifier Consensus"
        ... )
        >>> fig.savefig("forecast_consensus.png")
    """
    # Determine x-axis values (datetime index or sequential)
    if format_dates and hasattr(df.index, "to_pydatetime"):
        index = df.index
        use_dates = True
    else:
        index = range(len(df))
        use_dates = False

    with apply_nature_style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.1}
        )

        # ===== Top Panel: Eruption Probability =====
        if multi_model:
            # Plot individual classifier predictions
            for i, name in enumerate(model_names):
                col = f"{name}_eruption_probability"
                if col in df.columns:
                    ax1.plot(
                        index,
                        df[col],
                        linewidth=1.2,
                        alpha=0.6,
                        color=OKABE_ITO[i % len(OKABE_ITO)],
                        linestyle="--",
                        label=name.upper(),
                    )

            # Plot consensus with uncertainty band
            if "consensus_eruption_probability" in df.columns:
                cp = df["consensus_eruption_probability"]
                cu = df["consensus_uncertainty"]
                ax1.fill_between(
                    index,
                    (cp - cu).clip(0, 1),
                    (cp + cu).clip(0, 1),
                    alpha=0.25,
                    color=NATURE_COLORS["gray"],
                    label="Consensus ±1σ",
                )
                ax1.plot(
                    index,
                    cp,
                    color="black",
                    linewidth=2.5,
                    label="Consensus",
                    zorder=10,
                )
        else:
            # Single model: plot main probability
            prob_col = (
                f"{model_names[0]}_eruption_probability"
                if f"{model_names[0]}_eruption_probability" in df.columns
                else "eruption_probability"
            )
            ax1.plot(
                index,
                df[prob_col],
                color=OKABE_ITO[4],  # Blue
                linewidth=2.0,
                label="Eruption Probability",
            )

        # Add threshold line
        ax1.axhline(
            0.5,
            color=NATURE_COLORS["red"],
            linestyle="--",
            linewidth=1.5,
            label="Threshold (0.5)",
            alpha=0.7,
        )

        configure_spine(ax1)
        ax1.set_ylabel("Eruption Probability")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

        plot_title = "Eruption Forecast"
        if multi_model:
            plot_title += " — Multi-Model Consensus"
        if title:
            plot_title += f" ({title})"
        ax1.set_title(plot_title)

        # ===== Bottom Panel: Confidence =====
        if multi_model:
            # Plot individual classifier confidences
            for i, name in enumerate(model_names):
                col = f"{name}_confidence"
                if col in df.columns:
                    ax2.plot(
                        index,
                        df[col],
                        linewidth=1.2,
                        alpha=0.6,
                        color=OKABE_ITO[i % len(OKABE_ITO)],
                        linestyle="--",
                        label=name.upper(),
                    )

            # Plot consensus confidence
            if "consensus_confidence" in df.columns:
                ax2.plot(
                    index,
                    df["consensus_confidence"],
                    color="black",
                    linewidth=2.5,
                    label="Consensus",
                    zorder=10,
                )
        else:
            # Single model confidence
            conf_col = (
                f"{model_names[0]}_confidence"
                if f"{model_names[0]}_confidence" in df.columns
                else "confidence"
            )
            if conf_col in df.columns:
                ax2.plot(
                    index,
                    df[conf_col],
                    color=OKABE_ITO[4],  # Blue
                    linewidth=2.0,
                    label="Model Confidence",
                )

        # Add 0.5 reference line for confidence
        ax2.axhline(
            0.5,
            color=NATURE_COLORS["gray"],
            linestyle=":",
            linewidth=1.5,
            alpha=0.5,
        )

        configure_spine(ax2)
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

        # X-axis configuration
        if use_dates:
            # Format date axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(
                ax2.xaxis.get_majorticklabels(),
                rotation=30,
                horizontalalignment="right",
            )
            ax2.set_xlabel("Date")
        else:
            ax2.set_xlabel("Window Index")

    return fig


def plot_forecast_with_events(
    df: pd.DataFrame,
    model_names: list[str],
    eruption_dates: list[str] | None = None,
    multi_model: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 150,
) -> plt.Figure:
    """Plot forecast with eruption event markers overlay.

    Extends plot_forecast() by adding vertical red lines at known eruption dates
    for validation purposes. Useful for visually assessing model performance against
    historical events. Only eruptions within the DataFrame's date range are marked.

    Args:
        df (pd.DataFrame): Forecast DataFrame with datetime index. See plot_forecast()
            for required columns.
        model_names (list[str]): List of model names (keys for multi-model columns).
            For single model, pass list with one name.
        eruption_dates (list[str] | None, optional): List of eruption dates
            in "YYYY-MM-DD" format (e.g., ["2025-03-20", "2025-06-15"]).
            Only dates within df.index range are plotted. If None, no markers
            are added. Defaults to None.
        multi_model (bool, optional): Multi-model consensus mode. If True, shows
            individual classifiers and consensus. Defaults to False.
        title (str | None, optional): Plot title suffix appended to "Eruption Forecast".
            Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (12, 6).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with eruption markers overlaid on
            both probability and confidence subplots.

    Examples:
        >>> # Single model with eruption markers
        >>> fig = plot_forecast_with_events(
        ...     df, model_names=["xgb"],
        ...     eruption_dates=["2025-03-20", "2025-06-15"]
        ... )
        >>> fig.savefig("forecast_with_events.png")
        >>>
        >>> # Multi-model with validation events
        >>> fig = plot_forecast_with_events(
        ...     df, model_names=["rf", "xgb"],
        ...     eruption_dates=["2025-03-20"],
        ...     multi_model=True,
        ...     title="Validation Period"
        ... )
    """
    # Create base forecast plot
    fig = plot_forecast(
        df,
        model_names=model_names,
        multi_model=multi_model,
        title=title,
        figsize=figsize,
        dpi=dpi,
        format_dates=True,
    )

    # Add eruption event markers if provided
    if eruption_dates:
        axs = fig.get_axes()
        eruption_timestamps = pd.to_datetime(eruption_dates)

        for ax in axs:
            for eruption_date in eruption_timestamps:
                if df.index[0] <= eruption_date <= df.index[-1]:
                    ax.axvline(
                        eruption_date,
                        color=NATURE_COLORS["red"],
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.5,
                        zorder=5,
                    )
                    # Add label only to top axis
                    if ax == axs[0]:
                        ax.text(
                            eruption_date,
                            0.95,
                            f" Eruption\n {eruption_date.strftime('%Y-%m-%d')}",
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=8,
                            color=NATURE_COLORS["red"],
                        )

    return fig
