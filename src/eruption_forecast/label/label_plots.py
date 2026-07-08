import os
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import save_figure


class LabelScenarioEntry(TypedDict):
    """One scenario paired with an in-memory label DataFrame.

    Attributes:
        name (str): Human-readable scenario label shown on the X-axis (e.g.
            ``"Scenario 1.1"``).
        df (pd.DataFrame): DataFrame containing the label column to summarize.
    """

    name: str
    df: pd.DataFrame


class LabelScenarioFileEntry(TypedDict):
    """One scenario paired with the path to its LabelBuilder CSV.

    Attributes:
        name (str): Human-readable scenario label shown on the X-axis (e.g.
            ``"Scenario 1.1"``).
        csv (str): Path to a label CSV produced by
            :class:`~eruption_forecast.label.LabelBuilder`.
    """

    name: str
    csv: str


def plot_label_distribution(
    df: pd.DataFrame,
    filepath: str,
    *,
    label_column: str = "is_erupted",
    title: str | None = None,
    filetype: str = "png",
    color_non_erupted: str = "#1f77b4",
    color_erupted: str = "#d62728",
    figure_size: tuple[float, float] = (3.5, 2.625),
    dpi: int = 300,
    verbose: bool = True,
) -> str:
    """Plot class distribution as a bar chart with counts and percentages and save to disk.

    Produces a two-bar chart showing the number of samples in each class
    (``is_erupted = 0`` and ``is_erupted = 1``). Each bar is annotated with the
    raw count and the percentage relative to the total sample count. The figure
    is saved to ``filepath`` with the given ``filetype`` extension.

    Args:
        df (pd.DataFrame): DataFrame containing the label column to visualize.
        filepath (str): Destination path for the saved figure, without extension.
            The ``filetype`` extension is appended automatically by ``save_figure``.
        label_column (str): Column name holding binary class labels (0/1).
            Defaults to ``"is_erupted"``.
        title (str | None): Optional figure title. When ``None``, defaults to
            ``"Label Distribution"``.
        filetype (str): Image format (e.g. ``"png"``, ``"pdf"``). Defaults to
            ``"png"``.
        color_non_erupted (str): Hex color for the non-erupted (0) bar.
            Defaults to ``"#1f77b4"``.
        color_erupted (str): Hex color for the erupted (1) bar.
            Defaults to ``"#d62728"``.
        figure_size (tuple): Figure dimensions as ``(width, height)`` in inches.
            Defaults to ``(3.5, 2.625)``.
        dpi (int): Dots per inch for the saved figure. Defaults to ``300``.
        verbose (bool): Whether to log the saved filepath. Defaults to ``True``.

    Returns:
        str: Absolute path of the saved figure file (``filepath + "." + filetype``).

    Raises:
        KeyError: If ``label_column`` is not found in ``df``.
    """
    counts: pd.Series[Any] = df[label_column].value_counts().sort_index()
    total: int = int(counts.sum())

    class_labels = ["Non-erupted (0)", "Erupted (1)"]
    colors = [color_non_erupted, color_erupted]
    bar_values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    max_value = max(bar_values) if bar_values else 1

    fig, ax = plt.subplots(figsize=figure_size)

    bars = ax.bar(
        class_labels,
        bar_values,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, value in zip(bars, bar_values, strict=True):
        pct = value / total * 100 if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.02,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title if title is not None else "Label Distribution", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_ylim(0, max_value * 1.10)

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, filepath, dpi, filetype=filetype, verbose=verbose)
    saved_path = f"{filepath}.{filetype}"

    if verbose:
        logger.info(f"Label distribution plot saved to {saved_path}")

    return saved_path


def plot_label_distribution_from_file(
    label_csv: str,
    *,
    output_dir: str | None = None,
    label_column: str = "is_erupted",
    title: str | None = None,
    filetype: str = "png",
    color_non_erupted: str = "#1f77b4",
    color_erupted: str = "#d62728",
    figure_size: tuple[float, float] = (3.5, 2.625),
    dpi: int = 300,
    verbose: bool = True,
) -> str:
    """Plot label distribution from a LabelBuilder-generated CSV file.

    Loads the label CSV produced by :class:`~eruption_forecast.label.LabelBuilder`,
    then delegates to :func:`plot_label_distribution` to produce and save the bar
    chart.  The output figure is placed next to the CSV by default (using the same
    base filename with a ``_distribution`` suffix), or inside ``output_dir`` when
    that argument is provided.

    Args:
        label_csv (str): Path to the label CSV file generated by LabelBuilder.
        output_dir (str | None): Directory to save the figure in. When ``None``
            the figure is saved in the same directory as ``label_csv``.
            Defaults to ``None``.
        label_column (str): Column name holding binary class labels (0/1).
            Defaults to ``"is_erupted"``.
        title (str | None): Optional figure title. When ``None``, defaults to
            ``"Label Distribution"``.
        filetype (str): Image format (e.g. ``"png"``, ``"pdf"``). Defaults to
            ``"png"``.
        color_non_erupted (str): Hex color for the non-erupted (0) bar.
            Defaults to ``"#1f77b4"``.
        color_erupted (str): Hex color for the erupted (1) bar.
            Defaults to ``"#d62728"``.
        figure_size (tuple): Figure dimensions as ``(width, height)`` in inches.
            Defaults to ``(3.5, 2.625)``.
        dpi (int): Dots per inch for the saved figure. Defaults to ``300``.
        verbose (bool): Whether to log the saved filepath. Defaults to ``True``.

    Returns:
        str: Absolute path of the saved figure file.

    Raises:
        FileNotFoundError: If ``label_csv`` does not exist.
        KeyError: If ``label_column`` is not found in the loaded DataFrame.
    """
    if not os.path.isfile(label_csv):
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv, index_col=0)

    base_name = os.path.splitext(os.path.basename(label_csv))[0]
    save_dir = output_dir if output_dir is not None else os.path.dirname(label_csv)
    filepath = os.path.join(save_dir, f"{base_name}_distribution")

    return plot_label_distribution(
        df,
        filepath,
        label_column=label_column,
        title=title,
        filetype=filetype,
        color_non_erupted=color_non_erupted,
        color_erupted=color_erupted,
        figure_size=figure_size,
        dpi=dpi,
        verbose=verbose,
    )


def plot_label_distribution_comparison(
    entries: list[LabelScenarioEntry],
    filepath: str,
    *,
    label_column: str = "is_erupted",
    title: str | None = None,
    filetype: str = "png",
    color_non_erupted: str = "#1f77b4",
    color_erupted: str = "#d62728",
    figure_size: tuple[float, float] = (7.0, 3.5),
    dpi: int = 300,
    verbose: bool = True,
) -> str:
    """Plot a grouped bar chart comparing label distributions across scenarios.

    Renders one scenario group per ``entries`` element, each with two
    side-by-side bars (Non-erupted, Erupted). Bars are annotated with the raw
    count and the percentage relative to that scenario's own total, mirroring
    the annotation style of :func:`plot_label_distribution`. The figure is
    saved to ``filepath`` with the given ``filetype`` extension.

    Args:
        entries (list[LabelScenarioEntry]): Ordered list of scenarios. Each
            entry is a ``TypedDict`` with a ``name`` and an in-memory ``df``.
            The X-axis order matches the list order.
        filepath (str): Destination path for the saved figure, without
            extension. The ``filetype`` extension is appended automatically by
            ``save_figure``.
        label_column (str): Column name holding binary class labels (0/1).
            Defaults to ``"is_erupted"``.
        title (str | None): Optional figure title. When ``None``, defaults to
            ``"Label Distribution by Scenario"``.
        filetype (str): Image format (e.g. ``"png"``, ``"pdf"``). Defaults to
            ``"png"``.
        color_non_erupted (str): Hex color for the non-erupted (0) bars.
            Defaults to ``"#1f77b4"``.
        color_erupted (str): Hex color for the erupted (1) bars. Defaults to
            ``"#d62728"``.
        figure_size (tuple): Figure dimensions as ``(width, height)`` in inches.
            Defaults to ``(7.0, 3.5)``. Pass a wider width for large ``entries``
            counts to keep annotations legible.
        dpi (int): Dots per inch for the saved figure. Defaults to ``300``.
        verbose (bool): Whether to log the saved filepath. Defaults to ``True``.

    Returns:
        str: Absolute path of the saved figure file
            (``filepath + "." + filetype``).

    Raises:
        ValueError: If ``entries`` is empty.
        KeyError: If ``label_column`` is not found in any entry's DataFrame.
    """
    if not entries:
        raise ValueError("entries must contain at least one scenario")

    names: list[str] = []
    non_erupted_counts: list[int] = []
    erupted_counts: list[int] = []
    totals: list[int] = []

    for entry in entries:
        counts: pd.Series[Any] = entry["df"][label_column].value_counts().sort_index()
        non_erupted = int(counts.get(0, 0))
        erupted = int(counts.get(1, 0))
        names.append(entry["name"])
        non_erupted_counts.append(non_erupted)
        erupted_counts.append(erupted)
        totals.append(non_erupted + erupted)

    x = np.arange(len(names))
    width = 0.38
    max_value = max(non_erupted_counts + erupted_counts) or 1

    fig, ax = plt.subplots(figsize=figure_size)

    bars_non = ax.bar(
        x - width / 2,
        non_erupted_counts,
        width,
        color=color_non_erupted,
        edgecolor="white",
        linewidth=0.8,
        label="Non-erupted (0)",
    )
    bars_erupted = ax.bar(
        x + width / 2,
        erupted_counts,
        width,
        color=color_erupted,
        edgecolor="white",
        linewidth=0.8,
        label="Erupted (1)",
    )

    for bars, values in ((bars_non, non_erupted_counts), (bars_erupted, erupted_counts)):
        for bar, value, total in zip(bars, values, totals, strict=True):
            pct = value / total * 100 if total > 0 else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_value * 0.02,
                f"{value:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xticks(x, names)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(
        title if title is not None else "Label Distribution by Scenario",
        fontsize=9,
    )
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(0, max_value * 1.18)

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=7, frameon=False, loc="upper left")

    save_figure(fig, filepath, dpi, filetype=filetype, verbose=verbose)
    saved_path = f"{filepath}.{filetype}"

    if verbose:
        logger.info(f"Label distribution comparison plot saved to {saved_path}")

    return saved_path


def plot_label_distribution_comparison_from_files(
    entries: list[LabelScenarioFileEntry],
    filepath: str,
    *,
    label_column: str = "is_erupted",
    title: str | None = None,
    filetype: str = "png",
    color_non_erupted: str = "#1f77b4",
    color_erupted: str = "#d62728",
    figure_size: tuple[float, float] = (7.0, 3.5),
    dpi: int = 300,
    verbose: bool = True,
) -> str:
    """Plot a scenario comparison bar chart from LabelBuilder CSVs.

    Loads each entry's label CSV via ``pd.read_csv(csv, index_col=0)``, packs
    the results into :class:`LabelScenarioEntry` objects, and delegates to
    :func:`plot_label_distribution_comparison`. Existence of every CSV is
    checked up front so a missing file surfaces before any plotting work.

    Args:
        entries (list[LabelScenarioFileEntry]): Ordered list of scenarios,
            each a ``TypedDict`` with a ``name`` and a ``csv`` path.
        filepath (str): Destination path for the saved figure, without
            extension. The ``filetype`` extension is appended automatically by
            ``save_figure``.
        label_column (str): Column name holding binary class labels (0/1).
            Defaults to ``"is_erupted"``.
        title (str | None): Optional figure title. When ``None``, defaults to
            ``"Label Distribution by Scenario"``.
        filetype (str): Image format (e.g. ``"png"``, ``"pdf"``). Defaults to
            ``"png"``.
        color_non_erupted (str): Hex color for the non-erupted (0) bars.
            Defaults to ``"#1f77b4"``.
        color_erupted (str): Hex color for the erupted (1) bars. Defaults to
            ``"#d62728"``.
        figure_size (tuple): Figure dimensions as ``(width, height)`` in inches.
            Defaults to ``(7.0, 3.5)``.
        dpi (int): Dots per inch for the saved figure. Defaults to ``300``.
        verbose (bool): Whether to log the saved filepath. Defaults to ``True``.

    Returns:
        str: Absolute path of the saved figure file.

    Raises:
        ValueError: If ``entries`` is empty.
        FileNotFoundError: If any entry's ``csv`` path does not exist.
        KeyError: If ``label_column`` is not found in any loaded DataFrame.
    """
    if not entries:
        raise ValueError("entries must contain at least one scenario")

    df_entries: list[LabelScenarioEntry] = []
    for entry in entries:
        if not os.path.isfile(entry["csv"]):
            raise FileNotFoundError(f"Label CSV not found: {entry['csv']}")
        df_entries.append(
            {"name": entry["name"], "df": pd.read_csv(entry["csv"], index_col=0)}
        )

    return plot_label_distribution_comparison(
        df_entries,
        filepath,
        label_column=label_column,
        title=title,
        filetype=filetype,
        color_non_erupted=color_non_erupted,
        color_erupted=color_erupted,
        figure_size=figure_size,
        dpi=dpi,
        verbose=verbose,
    )
