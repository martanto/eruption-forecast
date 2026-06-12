"""SHAP-based explanation plots and dispatchers for ``ExplainerEnsemble``.

This module mirrors the dispatcher shape used by
:mod:`eruption_forecast.plots.evaluation_plots`: per-seed plots are looked up in
``PER_SEED_PLOT_DISPATCHER`` and module-level ``render_one_*`` workers handle
the actual save loop so ``joblib`` workers under the ``loky`` backend can
pickle them.
"""

import os
from typing import Literal
from collections.abc import Callable

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    nature_figure,
    configure_spine,
    apply_nature_style,
)
from eruption_forecast.utils.pathutils import save_figure


def plot_shap_bar(
    explanation: shap.Explanation,
    max_display: int = 20,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    title: str | None = None,
) -> plt.Figure:
    """Render a single-seed SHAP bar plot for the positive class.

    Wraps :func:`shap.plots.bar` inside the project's Nature-style figure
    context so the resulting PNG matches the rest of the pipeline's plots.

    Args:
        explanation (shap.Explanation): Per-seed SHAP explanation. Already
            sliced to the positive class by the worker.
        max_display (int, optional): Maximum features to display. Defaults
            to ``20``.
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to ``(8, 6)``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Defaults to ``None``.

    Returns:
        plt.Figure: The rendered matplotlib figure.
    """
    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        shap.plots.bar(explanation, max_display=max_display, show=False, ax=ax)
        if title:
            ax.set_title(title)
        configure_spine(ax)
    return fig


def plot_shap_beeswarm(
    explanation: shap.Explanation,
    max_display: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
    dot_size: float = 32.0,
) -> plt.Figure:
    """Render a single-seed SHAP beeswarm plot for the positive class.

    Always forwards ``plot_size=None`` to :func:`shap.plots.beeswarm` so the
    pre-created figure size is respected, per the project's standing SHAP
    rule. The figure is created via :func:`matplotlib.pyplot.figure` (no
    pre-built axes) so SHAP can lay out its own main + colorbar axes; the
    title is then centered horizontally on the main axes via
    :meth:`fig.suptitle` because long tsfresh feature names push the
    SHAP-drawn axes off-center inside the figure.

    Args:
        explanation (shap.Explanation): Per-seed SHAP explanation. Already
            sliced to the positive class by the worker.
        max_display (int, optional): Maximum features to display. Defaults
            to ``20``.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes to ``(16, max(8, max_display * 0.5))``
            so long y-tick labels remain readable. Defaults to ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Falls back to ``"SHAP
            Summary Plot"`` when omitted. Defaults to ``None``.
        dot_size (float, optional): Marker size for individual SHAP dots,
            forwarded to ``shap.plots.beeswarm(s=...)``. SHAP's default is
            ``16``; bumped here so dots remain visible inside the wider
            figure. Defaults to ``32.0``.

    Returns:
        plt.Figure: The rendered matplotlib figure.
    """
    fig_size: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with apply_nature_style():
        fig = plt.figure(figsize=fig_size, dpi=dpi)

        shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            s=dot_size,
            plot_size=None,
            show=False,
        )

        #  SHAP draws into the current axes and then adds a colorbar axes,
        #  so ``fig.axes[0]`` is the beeswarm panel itself. Center the title
        #  on it rather than on the whole figure — long feature labels push
        #  the panel rightward and a figure-centered title looks misaligned.
        ax = fig.axes[0]
        pos = ax.get_position()
        fig.suptitle(
            title or "SHAP Summary Plot",
            x=(pos.x0 + pos.x1) / 2,
            y=0.9,
            ha="center",
        )
        configure_spine(ax)
    return fig


def plot_shap_waterfall(
    explanation_one: shap.Explanation,
    max_display: int = 15,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    title: str | None = None,
) -> plt.Figure:
    """Render a single-observation SHAP waterfall plot.

    Args:
        explanation_one (shap.Explanation): Single-row SHAP explanation —
            typically ``explanation[i]`` for the i-th observation.
        max_display (int, optional): Maximum features to display. Defaults
            to ``15``.
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to ``(8, 6)``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Defaults to ``None``.

    Returns:
        plt.Figure: The rendered matplotlib figure.
    """
    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        shap.plots.waterfall(
            explanation_one,
            max_display=max_display,
            show=False,
        )
        if title:
            ax.set_title(title)
    return fig


def plot_aggregate_shap_bar(
    aggregate_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Render a frequency-weighted aggregate SHAP bar plot.

    Expects an ``aggregate_df`` produced by
    :meth:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble._aggregate_importance`,
    containing columns ``feature``, ``mean_abs_shap``, ``selection_frequency``,
    ``n_seeds_selected``. Bars are ranked by ``mean_abs_shap``; the
    ``selection_frequency`` is overlaid as a right-edge text annotation so
    callers can spot features that were highly impactful but selected by only
    a few seeds.

    Args:
        aggregate_df (pd.DataFrame): Aggregate importance DataFrame.
        top_n (int, optional): Number of top features to display. Defaults
            to ``20``.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes to ``(8, max(4, top_n * 0.35))``.
            Defaults to ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Defaults to ``None``.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: The figure and the unmodified
            ``aggregate_df`` (so :func:`render_one_aggregate_plot` can save
            it as the CSV companion).
    """
    df = aggregate_df.head(top_n).copy()
    figheight = max(4.0, top_n * 0.35)
    fig_size: tuple[float, float] = figsize if figsize is not None else (8.0, figheight)

    #  Reverse so the most important feature ends up at the top of the bar
    #  chart — matplotlib draws ``barh`` bottom-up.
    df_plot = df.iloc[::-1].reset_index(drop=True)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        bars = ax.barh(
            range(len(df_plot)),
            df_plot["mean_abs_shap"],
            color=OKABE_ITO[4],
            alpha=0.85,
        )
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["feature"].tolist())

        for i, (_bar, mean_val, freq) in enumerate(
            zip(bars, df_plot["mean_abs_shap"], df_plot["selection_frequency"], strict=True)
        ):
            ax.text(
                mean_val,
                i,
                f"  {mean_val:.3f}  (sel={freq:.0%})",
                va="center",
                ha="left",
                fontsize=7,
                color=NATURE_COLORS["blue"],
            )

        configure_spine(ax)
        ax.set_xlabel("Mean |SHAP| (frequency-weighted)")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"Aggregate Top-{top_n} SHAP Importances")

    return fig, aggregate_df


def plot_aggregate_shap_beeswarm(
    explanation: shap.Explanation,
    row_seed: list[int],
    row_obs: list[int],
    max_display: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
    dot_size: float = 32.0,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Render a stacked-seeds aggregate SHAP beeswarm.

    Consumes the union-of-features Explanation produced by
    :meth:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble._aggregate_explanation`.
    Cells where a seed did not select a feature carry NaN and are skipped by
    SHAP's beeswarm internals. Returns a tidy long-form sidecar DataFrame so
    external tools can redraw the swarm without the worker pickle.

    Args:
        explanation (shap.Explanation): Stacked positive-class explanation
            with shape ``(n_seeds × n_obs, |union features|)``.
        row_seed (list[int]): Seed ``random_state`` for each row of
            ``explanation.values``.
        row_obs (list[int]): Window id for each row of
            ``explanation.values``.
        max_display (int, optional): Maximum features to display. Defaults
            to ``20``.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes to ``(16, max(8, max_display * 0.5))``.
            Defaults to ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Falls back to ``"SHAP
            Summary Plot"`` when omitted. Defaults to ``None``.
        dot_size (float, optional): Marker size for individual SHAP dots,
            forwarded to ``shap.plots.beeswarm(s=...)``. SHAP's default is
            ``16``; bumped here so dots remain visible inside the wider
            figure. Defaults to ``32.0``.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: The figure and a tidy DataFrame
            with columns ``feature``, ``random_state``, ``obs_id``,
            ``shap_value``, ``feature_value`` covering only the non-NaN
            cells.
    """
    fig_size: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with apply_nature_style():
        fig = plt.figure(figsize=fig_size, dpi=dpi)

        shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            s=dot_size,
            plot_size=None,
            show=False,
        )

        #  SHAP draws into the current axes and then adds a colorbar axes,
        #  so ``fig.axes[0]`` is the beeswarm panel itself. Center the title
        #  on it rather than on the whole figure — long feature labels push
        #  the panel rightward and a figure-centered title looks misaligned.
        ax = fig.axes[0]
        pos = ax.get_position()
        fig.suptitle(
            title or "SHAP Summary Plot",
            x=(pos.x0 + pos.x1) / 2,
            y=0.9,
            ha="center",
        )
        configure_spine(ax)

    #  Tidy long-form sidecar: one row per non-NaN cell. Lets a downstream
    #  consumer rebuild the swarm or query a specific (seed, obs, feature)
    #  triple without re-loading the explanation pickle.
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    feature_names = list(explanation.feature_names)

    seed_arr = np.asarray(row_seed)
    obs_arr = np.asarray(row_obs)

    tidy_rows: list[dict] = []
    for col_index, feature in enumerate(feature_names):
        col_values = values[:, col_index]
        col_data = data[:, col_index]
        mask = ~np.isnan(col_values)
        if not mask.any():
            continue
        for shap_value, feature_value, seed_value, obs_value in zip(
            col_values[mask],
            col_data[mask],
            seed_arr[mask],
            obs_arr[mask],
            strict=True,
        ):
            tidy_rows.append(
                {
                    "feature": feature,
                    "random_state": int(seed_value),
                    "obs_id": int(obs_value),
                    "shap_value": float(shap_value),
                    "feature_value": float(feature_value),
                }
            )

    tidy_df = pd.DataFrame(
        tidy_rows,
        columns=["feature", "random_state", "obs_id", "shap_value", "feature_value"],
    )
    return fig, tidy_df


PER_SEED_PLOT_DISPATCHER: dict[str, Callable[..., plt.Figure]] = {
    "bar": plot_shap_bar,
    "beeswarm": plot_shap_beeswarm,
}


AGGREGATE_PLOT_DISPATCHER: dict[str, Callable[..., tuple[plt.Figure, pd.DataFrame]]] = {
    "bar": plot_aggregate_shap_bar,
    "beeswarm": plot_aggregate_shap_beeswarm,
}


#  Each aggregate plot consumes a different input shape — the bar takes the
#  feature-level summary DataFrame, the beeswarm takes the stacked
#  ``shap.Explanation``. This sibling registry tells the orchestrator and the
#  render worker which builder to invoke for each plot.
AGGREGATE_PLOT_INPUT_KIND: dict[str, Literal["dataframe", "explanation"]] = {
    "bar": "dataframe",
    "beeswarm": "explanation",
}


def render_one_seed_plot(
    classifier_name: str,
    random_state: int,
    plot_name: str,
    shap_values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    dpi: int = 150,
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    """Render and persist a single per-seed SHAP plot.

    Module-level so ``joblib`` workers under the ``loky`` backend can pickle
    it. Builds a :class:`shap.Explanation` on the fly from the persisted
    per-seed arrays and dispatches to the function registered in
    :data:`PER_SEED_PLOT_DISPATCHER`. Saves to
    ``{output_dir}/{classifier_name}/figures/{plot_name}/{random_state:05d}.png``.

    Args:
        classifier_name (str): Classifier identifier used as the first path
            segment under ``output_dir``.
        random_state (int): Seed value, formatted to a zero-padded 5-digit
            file stem.
        plot_name (str): Key into :data:`PER_SEED_PLOT_DISPATCHER`.
        shap_values (np.ndarray): Positive-class SHAP values of shape
            ``(n_obs, n_features_seed)``.
        base_value (float): Positive-class baseline (scalar).
        feature_values (np.ndarray): Raw feature values of shape
            ``(n_obs, n_features_seed)``, matched to ``feature_names``.
        feature_names (list[str]): Feature names for this seed.
        output_dir (str): Root directory for figure output.
        dpi (int, optional): Figure resolution. Defaults to ``150``.
        overwrite (bool, optional): When ``True``, regenerate the figure
            even if a ``.png`` already exists. Defaults to ``False``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        str: Filepath stem (without extension) written by
            :func:`save_figure`.
    """
    plot_function = PER_SEED_PLOT_DISPATCHER[plot_name]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], base_value),
        data=feature_values,
        feature_names=feature_names,
    )

    fig = plot_function(explanation)

    filepath = os.path.join(
        output_dir,
        classifier_name,
        "figures",
        plot_name,
        f"{random_state:05d}",
    )

    if not overwrite and os.path.exists(f"{filepath}.png"):
        if verbose:
            logger.info(f"[SHAP Seed Plot/{plot_name}/{random_state:05d}] exists.")
        return filepath

    save_figure(fig, filepath, dpi, verbose=verbose)
    return filepath


def render_one_waterfall_plot(
    classifier_name: str,
    random_state: int,
    obs_index: int,
    obs_id: int,
    shap_values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    dpi: int = 150,
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    """Render and persist a single-observation SHAP waterfall plot.

    Module-level so ``joblib`` workers under the ``loky`` backend can pickle
    it. Saves to
    ``{output_dir}/{classifier_name}/figures/waterfall/{random_state:05d}_{obs_id:05d}.png``.

    Args:
        classifier_name (str): Classifier identifier used as the first path
            segment under ``output_dir``.
        random_state (int): Seed value.
        obs_index (int): Position of the observation inside the seed's
            ``shap_values`` matrix.
        obs_id (int): The window id of the explained observation; used in
            the output filename so the file can be matched back to the
            original window without re-loading the SHAP pickle.
        shap_values (np.ndarray): Positive-class SHAP values of shape
            ``(n_obs, n_features_seed)``.
        base_value (float): Positive-class baseline (scalar).
        feature_values (np.ndarray): Raw feature values of shape
            ``(n_obs, n_features_seed)``.
        feature_names (list[str]): Feature names for this seed.
        output_dir (str): Root directory for figure output.
        dpi (int, optional): Figure resolution. Defaults to ``150``.
        overwrite (bool, optional): When ``True``, regenerate the figure
            even if a ``.png`` already exists. Defaults to ``False``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        str: Filepath stem (without extension) written by
            :func:`save_figure`.
    """
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], base_value),
        data=feature_values,
        feature_names=feature_names,
    )

    fig = plot_shap_waterfall(explanation[obs_index])

    filepath = os.path.join(
        output_dir,
        classifier_name,
        "figures",
        "waterfall",
        f"{random_state:05d}_{obs_id:05d}",
    )

    if not overwrite and os.path.exists(f"{filepath}.png"):
        if verbose:
            logger.info(
                f"[SHAP Waterfall/{random_state:05d}/{obs_id:05d}] exists."
            )
        return filepath

    save_figure(fig, filepath, dpi, verbose=verbose)
    return filepath


def render_one_aggregate_plot(
    classifier_name: str,
    plot_name: str,
    aggregate_input: pd.DataFrame | tuple[shap.Explanation, list[int], list[int]],
    output_dir: str,
    dpi: int = 150,
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    """Render and persist a single aggregate SHAP plot.

    Module-level so ``joblib`` workers under the ``loky`` backend can pickle
    it. Saves to
    ``{output_dir}/{classifier_name}/figures/aggregate/{plot_name}.{png,csv}``.
    Dispatches the second-argument shape on
    :data:`AGGREGATE_PLOT_INPUT_KIND` so each plot receives the input type
    it actually needs (feature-level DataFrame for the bar, stacked
    ``shap.Explanation`` for the beeswarm).

    Args:
        classifier_name (str): Classifier identifier used as the first path
            segment under ``output_dir``.
        plot_name (str): Key into :data:`AGGREGATE_PLOT_DISPATCHER`.
        aggregate_input (pd.DataFrame | tuple[shap.Explanation, list[int],
            list[int]]): Plot-specific input. ``"dataframe"`` plots take the
            aggregate-importance DataFrame produced by
            :meth:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble._aggregate_importance`;
            ``"explanation"`` plots take the
            ``(stacked_explanation, row_seed, row_obs)`` tuple produced by
            :meth:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble._aggregate_explanation`.
        output_dir (str): Root directory for figure output.
        dpi (int, optional): Figure resolution. Defaults to ``150``.
        overwrite (bool, optional): When ``True``, regenerate the figure
            and CSV even if both already exist. Defaults to ``False``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        str: Filepath stem (without extension). The PNG figure and the CSV
            data table share this stem.
    """
    plot_function = AGGREGATE_PLOT_DISPATCHER[plot_name]
    input_kind = AGGREGATE_PLOT_INPUT_KIND[plot_name]

    if input_kind == "dataframe":
        fig, data = plot_function(aggregate_input, dpi=dpi)
    else:
        explanation, row_seed, row_obs = aggregate_input
        fig, data = plot_function(
            explanation, row_seed=row_seed, row_obs=row_obs, dpi=dpi
        )

    filepath = os.path.join(
        output_dir,
        classifier_name,
        "figures",
        "aggregate",
        plot_name,
    )

    if (
        not overwrite
        and os.path.exists(f"{filepath}.png")
        and os.path.exists(f"{filepath}.csv")
    ):
        if verbose:
            logger.info(f"[SHAP Aggregate Plot/{plot_name}] exists.")
        return filepath

    save_figure(fig, filepath, dpi, verbose=verbose)
    data.to_csv(f"{filepath}.csv", index=False)
    return filepath
