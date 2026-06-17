import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.ensemble import ClassifierEnsemble
from eruption_forecast.utils.ml import build_classifier_ensemble_summary
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    shap_figure,
    configure_spine,
    apply_nature_style,
)
from eruption_forecast.utils.pathutils import save_figure
from eruption_forecast.dataclass.classifier_explanation import ClassifierExplanation
from eruption_forecast.dataclass.classifier_ensemble_summary import (
    ClassifierEnsembleSummary,
)


def plot_shap_beeswarm(
    explanation: shap.Explanation,
    max_display: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_filepath: str | None = None,
    dpi: int = 150,
    verbose: bool = True,
    group_remaining_features: bool = False,
) -> plt.Figure:
    fig_size: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=fig_size, dpi=dpi) as fig:
        shap.plots.beeswarm(
            shap_values=(
                explanation[..., 1] if explanation.values.ndim == 3 else explanation
            ),
            max_display=max_display,
            group_remaining_features=group_remaining_features,
            s=32,
            plot_size=None,
            show=False,
        )

        #  SHAP draws into the current axes and then adds a colorbar axes,
        #  so ``fig.axes[0]`` is the beeswarm panel itself. Center the title
        #  on it rather than on the whole figure; long feature labels push
        #  the panel rightward and a figure-centered title looks misaligned.
        ax = fig.axes[0]
        ax.set_xlabel("Mean SHAP value", fontsize=12)
        pos = ax.get_position()

        fig.suptitle(
            title or "SHAP Summary Plot",
            x=(pos.x0 + pos.x1) / 2,
            y=0.9,
            ha="center",
        )

        if save_filepath:
            save_figure(fig, save_filepath, dpi, verbose=verbose)

    return fig


def plot_shap_bar(
    explanation: shap.Explanation,
    max_display: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_filepath: str | None = None,
    dpi: int = 150,
    verbose: bool = True,
) -> plt.Figure:
    fig_size: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=fig_size, dpi=dpi) as fig:
        shap.plots.bar(
            explanation,
            max_display=max_display,
            show=False,
        )

        fig.suptitle(title or "SHAP Bar Plot", y=0.9)

        if save_filepath:
            save_figure(fig, save_filepath, dpi, verbose=verbose)

    return fig


def render_seed_plot(
    plot_kind: str,
    explanation: shap.Explanation,
    save_filepath: str,
    title: str,
    max_display: int,
    group_remaining_features: bool,
    dpi: int,
    verbose: bool,
) -> str:
    """Render and save one per-seed SHAP plot.

    Module-level so ``joblib`` workers under the ``loky`` backend can
    pickle it. Dispatches on ``plot_kind`` to the matching renderer.

    Args:
        plot_kind: ``"beeswarm"`` or ``"bar"``.
        explanation: Per-seed SHAP explanation.
        save_filepath: Destination ``.png`` path; closed by ``save_figure``.
        title: Suptitle for the rendered plot.
        max_display: Maximum number of features to display.
        group_remaining_features: Forwarded to ``shap.plots.beeswarm``.
        dpi: Figure resolution.
        verbose: Forwarded to ``save_figure``.

    Returns:
        str: ``save_filepath``.
    """
    if plot_kind == "beeswarm":
        plot_shap_beeswarm(
            explanation=explanation,
            max_display=max_display,
            title=title,
            save_filepath=save_filepath,
            dpi=dpi,
            verbose=verbose,
            group_remaining_features=group_remaining_features,
        )
    elif plot_kind == "bar":
        plot_shap_bar(
            explanation=explanation,
            max_display=max_display,
            title=title,
            save_filepath=save_filepath,
            dpi=dpi,
            verbose=verbose,
        )
    return save_filepath


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
            zip(
                bars,
                df_plot["mean_abs_shap"],
                df_plot["selection_frequency"],
                strict=True,
            )
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


def plot_shap_waterfall(
    eruption_dates: list[str],
    labels: pd.Series | pd.DataFrame,
    classifier_explanation: ClassifierExplanation,
    classifier_ensemble: ClassifierEnsemble,
):
    classifier_name = classifier_explanation.classifier_name
    seed_ensemble = classifier_ensemble[classifier_name]

    classifier_ensemble_summary = build_classifier_ensemble_summary(
        seed_ensemble=seed_ensemble,
        labels=labels,
        eruption_dates=eruption_dates,
    )

    return None


def plot_classifier_waterfall(
    summary: ClassifierEnsembleSummary,
    classifier_explanation: ClassifierExplanation,
    max_display: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Render the SHAP waterfall for the top pick on ``summary``.

    Looks up the seed identified by
    :attr:`ClassifierEnsembleSummary.highest.random_state` inside
    ``classifier_explanation.seeds`` (by ``random_state``, not list
    position), then renders ``shap.plots.waterfall`` for the corresponding
    explanation row.

    Args:
        summary (ClassifierEnsembleSummary): Summary returned by
            :func:`eruption_forecast.utils.ml.build_classifier_ensemble_summary`.
        classifier_explanation (ClassifierExplanation): One of the dataclass
            payloads produced by :class:`ExplainerEnsemble.explain`. Its
            ``seeds`` attribute is a list of ``SeedExplanation`` records
            exposing ``random_state`` and ``shap_values``.
        max_display (int, optional): Maximum number of features in the
            waterfall. Defaults to 20.
        figsize (tuple[float, float] | None, optional): Figure size in inches.
            When ``None`` uses ``(16.0, max(8.0, max_display * 0.5))`` so long
            tsfresh labels remain readable. Defaults to ``None``.
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: The figure containing the rendered waterfall.

    Raises:
        RuntimeError: If ``summary.highest`` is ``None`` — no eruption window
            intersected the prediction grid, so there is no row to render.
        KeyError: If the seed identified by ``summary.highest.random_state`` is
            absent from ``classifier_explanation.seeds``.
    """
    if summary.highest is None:
        raise RuntimeError(
            f"{summary.classifier_name}: no eruption windows intersected the "
            f"prediction grid; nothing to render."
        )

    seeds_by_random_state = {
        int(seed.random_state): seed for seed in classifier_explanation.seeds
    }
    seed = seeds_by_random_state[summary.highest.random_state]
    explanation: shap.Explanation = seed.shap_values[summary.highest.index]

    fig_size = figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))

    with shap_figure(figsize=fig_size, dpi=dpi) as fig:
        shap.plots.waterfall(
            shap_values=explanation,
            max_display=max_display,
            show=False,
        )
        fig.suptitle(
            f"{summary.classifier_name} "
            f"[s={summary.highest.random_state}|p={summary.highest.value:.4f}]\n"
            f"{summary.highest.datetime:%Y-%m-%d %H:%M:%S}",
            fontsize=12,
        )

    return fig
