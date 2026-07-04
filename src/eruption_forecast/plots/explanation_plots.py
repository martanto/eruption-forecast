import os

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.ensemble import ClassifierEnsemble
from eruption_forecast.utils.ml import build_classifier_ensemble_summary
from eruption_forecast.dataclass import EruptionWindow
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    shap_figure,
    configure_spine,
)
from eruption_forecast.utils.pathutils import save_figure
from eruption_forecast.dataclass.classifier_explanation import ClassifierExplanation


def plot_shap_waterfall(
    explanation: shap.Explanation,
    max_display: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_filepath: str | None = None,
    dpi: int = 150,
    verbose: bool = False,
) -> plt.Figure:
    """Render a single SHAP waterfall plot for one observation.

    Wraps ``shap.plots.waterfall`` inside the project's
    :func:`shap_figure` context manager so the figure size and DPI are set
    before SHAP draws.

    Args:
        explanation (shap.Explanation): Single-row SHAP explanation to
            visualise.
        max_display (int, optional): Maximum number of features to
            display. Defaults to ``20``.
        title (str | None, optional): Plot suptitle. ``None`` leaves the
            figure title blank. Defaults to ``None``.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes to ``(16, max(8, max_display *
            0.5))``. Defaults to ``None``.
        save_filepath (str | None, optional): Destination path. ``None``
            skips saving. Defaults to ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        plt.Figure: The rendered figure.
    """
    figsize: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=figsize, dpi=150) as fig:
        shap.plots.waterfall(
            shap_values=explanation,
            max_display=max_display,
            show=False,
        )

        fig.suptitle(title, y=0.9)

        if save_filepath:
            save_figure(fig, save_filepath, dpi, verbose=verbose)

    return fig


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
    figsize: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=figsize, dpi=dpi) as fig:
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

    plt.close(fig)

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
    figsize: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=figsize, dpi=dpi) as fig:
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


def _aggregate_importance(
    classifier_explanation: ClassifierExplanation,
) -> pd.DataFrame:
    """Compute a per-feature aggregate importance table across all seeds.

    For each feature appearing in any seed's ``shap.Explanation``, records
    the mean ``|SHAP|`` averaged over the seeds that selected the feature
    (the conditional importance), the count of seeds that selected it, and
    the corresponding ``selection_frequency``. Seed values are reduced
    from 3D to 2D by slicing the positive class (mirrors the per-seed
    beeswarm rendering at :func:`plot_shap_beeswarm`) before the per-seed
    mean ``|SHAP|`` is taken.

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            bundle of per-seed ``SeedExplanation`` payloads.

    Returns:
        pd.DataFrame: Frequency-weighted importance table sorted by
            ``mean_abs_shap`` descending. Columns are ``feature``,
            ``mean_abs_shap``, ``selection_frequency``, and
            ``n_seeds_selected``. The schema matches the input contract
            of :func:`plot_aggregate_shap_bar`.
    """
    n_total_seeds = len(classifier_explanation.seeds)
    accumulator: dict[str, list[float]] = {}
    for seed in classifier_explanation.seeds:
        explanation = seed.shap_values
        values = np.asarray(explanation.values)  # noqa: PD011
        if values.ndim == 3:
            values = values[..., 1]
        mean_abs = np.abs(values).mean(axis=0)
        for name, val in zip(list(explanation.feature_names), mean_abs, strict=True):
            accumulator.setdefault(name, []).append(float(val))

    rows = [
        {
            "feature": feature,
            "mean_abs_shap": float(np.mean(seed_values)),
            "selection_frequency": len(seed_values) / n_total_seeds,
            "n_seeds_selected": len(seed_values),
        }
        for feature, seed_values in accumulator.items()
    ]
    importance_df = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "mean_abs_shap",
            "selection_frequency",
            "n_seeds_selected",
        ],
    )
    return importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )


def _aggregate_explanation(
    classifier_explanation: ClassifierExplanation,
) -> tuple[shap.Explanation, list[int], list[int]]:
    """Stack per-seed ``shap.Explanation``s into a union feature space.

    Builds the ordered union of feature names across every seed
    (preserving first-seen order so re-runs are deterministic), then
    NaN-pads each seed's ``values`` and ``data`` matrices from its
    per-seed columns into the union space and concatenates along the
    sample axis. Cells where a seed did not select a feature carry
    ``np.nan`` so SHAP's beeswarm internals skip them — preserving the
    "this seed didn't pick it" signal instead of inflating the swarm
    with fake zeros. 3D ``values`` are reduced to the positive class
    before padding (matches the per-seed beeswarm path at
    :func:`plot_shap_beeswarm`).

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            bundle of per-seed ``SeedExplanation`` payloads.

    Returns:
        tuple[shap.Explanation, list[int], list[int]]: A 3-tuple of:

            - **explanation** (``shap.Explanation``): Stacked explanation
              of shape ``(n_seeds * n_obs, |union features|)`` with
              ``feature_names`` set to the union list.
            - **row_seed** (``list[int]``): ``random_state`` for each
              row of ``explanation.values``.
            - **row_obs** (``list[int]``): Window id for each row of
              ``explanation.values`` (the row index within the
              per-seed explanation, since per-seed explanations are
              aligned positionally to ``features_df``).
    """
    all_names: list[str] = []
    seen: set[str] = set()
    for seed in classifier_explanation.seeds:
        for name in seed.shap_values.feature_names:
            if name not in seen:
                seen.add(name)
                all_names.append(name)
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    n_features = len(all_names)

    values_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    row_seed: list[int] = []
    row_obs: list[int] = []

    for seed in classifier_explanation.seeds:
        explanation = seed.shap_values
        values = np.asarray(explanation.values)  # noqa: PD011
        if values.ndim == 3:
            values = values[..., 1]
        raw_data = getattr(explanation, "data", None)
        data = (
            np.asarray(raw_data)
            if raw_data is not None
            else np.full_like(values, np.nan)
        )
        n_samples = values.shape[0]

        padded_values = np.full((n_samples, n_features), np.nan)
        padded_data = np.full((n_samples, n_features), np.nan)
        for j, name in enumerate(explanation.feature_names):
            col_idx = name_to_idx[name]
            padded_values[:, col_idx] = values[:, j]
            padded_data[:, col_idx] = data[:, j]

        values_blocks.append(padded_values)
        data_blocks.append(padded_data)
        row_seed.extend([int(seed.random_state)] * n_samples)
        row_obs.extend(range(n_samples))

    merged_values = np.concatenate(values_blocks, axis=0)
    merged_data = np.concatenate(data_blocks, axis=0)

    aggregated = shap.Explanation(
        values=merged_values,
        data=merged_data,
        feature_names=all_names,
    )
    return aggregated, row_seed, row_obs


def plot_aggregate_shap_bar(
    classifier_explanation: ClassifierExplanation,
    top_n: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
    save_filepath: str | None = None,
    verbose: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Render a frequency-weighted aggregate SHAP bar plot.

    Builds the per-feature aggregate importance table via
    :func:`_aggregate_importance` (columns ``feature``, ``mean_abs_shap``,
    ``selection_frequency``, ``n_seeds_selected``), then renders the
    top-``top_n`` rows as a horizontal bar chart. Bars are ranked by
    ``mean_abs_shap``; the ``selection_frequency`` is overlaid as a
    right-edge text annotation so callers can spot features that were
    highly impactful but selected by only a few seeds.

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            bundle of per-seed ``SeedExplanation`` payloads.
        top_n (int, optional): Number of top features to display. Defaults
            to ``20``.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes to ``(8, max(4, top_n * 0.35))``.
            Defaults to ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        title (str | None, optional): Plot title. Defaults to ``None``.
        save_filepath (str | None, optional): Destination path. ``None``
            skips saving. Defaults to ``None``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: The figure and the full
            importance DataFrame (not truncated to ``top_n``) so the
            caller can persist it as the CSV companion.
    """
    importance_df = _aggregate_importance(classifier_explanation)

    df = importance_df.head(top_n).copy()
    figheight = max(4.0, top_n * 0.35)
    figsize: tuple[float, float] = figsize if figsize is not None else (8.0, figheight)

    #  Reverse so the most important feature ends up at the top of the bar
    #  chart — matplotlib draws ``barh`` bottom-up.
    df_plot = df.iloc[::-1].reset_index(drop=True)

    with shap_figure(figsize=figsize, dpi=dpi) as fig:
        ax = fig.add_subplot(111)

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
                f"  {mean_val:.3f} (sel={freq:.0%})",  # sel=80%, means 80% of seeds selected that feature
                va="center",
                ha="left",
                fontsize=7,
                color=NATURE_COLORS["blue"],
            )

        configure_spine(ax)
        ax.set_xlabel("Mean |SHAP| (frequency-weighted)")
        ax.set_ylabel("Feature")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title or f"Aggregate Top-{top_n} SHAP Importances")

        if save_filepath:
            save_figure(fig, save_filepath, dpi, verbose=verbose)

    return fig, importance_df


def plot_aggregate_shap_beeswarm(
    classifier_explanation: ClassifierExplanation,
    max_display: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
    dot_size: float = 32.0,
    group_remaining_features: bool = False,
    save_filepath: str | None = None,
    verbose: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Render a stacked-seeds aggregate SHAP beeswarm.

    Builds a NaN-padded union-of-features Explanation via
    :func:`_aggregate_explanation` and renders it as a single beeswarm.
    Cells where a seed did not select a feature carry NaN and are skipped
    by SHAP's beeswarm internals. Returns a tidy long-form sidecar
    DataFrame so external tools can redraw the swarm without the worker
    pickle.

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            bundle of per-seed ``SeedExplanation`` payloads.
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
        group_remaining_features (bool, optional): Forwarded to
            ``shap.plots.beeswarm``. Defaults to ``False``.
        save_filepath (str | None, optional): Destination path. ``None``
            skips saving. Defaults to ``None``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: The figure and a tidy DataFrame
            with columns ``feature``, ``random_state``, ``obs_id``,
            ``shap_value``, ``feature_value`` covering only the non-NaN
            cells.
    """
    explanation, row_seed, row_obs = _aggregate_explanation(classifier_explanation)

    figsize: tuple[float, float] = (
        figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))
    )

    with shap_figure(figsize=figsize, dpi=dpi) as fig:
        shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            s=dot_size,
            plot_size=None,
            show=False,
            group_remaining_features=group_remaining_features,
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

        if save_filepath:
            save_figure(fig, save_filepath, dpi, verbose=verbose)

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


def _build_explanation_plot(
    classifier_explanation: ClassifierExplanation,
    eruption_window: EruptionWindow,
    eruption_dir: str,
    classifier_name: str,
) -> tuple[shap.Explanation, str, str]:
    """Pick the highest-probability window for one eruption and prepare its plot inputs.

    Resolves the seed index and window index of the highest-probability
    observation for a given eruption, then derives a deterministic save
    path and figure title from those identifiers.

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            SHAP explanations indexed by ``random_state``.
        eruption_window (EruptionWindow): Eruption summary carrying the
            ``highest`` ``ProbabilityPick`` used to locate the SHAP row.
        eruption_dir (str): Root directory under which a
            ``{eruption_date}/`` subdirectory is created for the saved
            figure.
        classifier_name (str): Classifier identifier used in the filename
            and figure title.

    Returns:
        tuple[shap.Explanation, str, str]: ``(explanation, save_filepath,
            title)`` ready for :func:`plot_shap_waterfall`.
    """
    eruption_date = eruption_window.eruption_date
    eruption_date_dir = os.path.join(eruption_dir, eruption_date)

    seed_idx = eruption_window.highest.random_state
    eruption_window_index = eruption_window.highest.index
    proba_value = eruption_window.highest.value
    eruption_datetime = f"{eruption_window.highest.datetime:%Y-%m-%d_%H-%M-%S}"

    filename = f"{classifier_name}_{eruption_datetime}_seed={seed_idx}_index={eruption_window_index}.png"
    save_filepath = os.path.join(eruption_date_dir, filename)

    explanation = classifier_explanation.seeds[seed_idx].shap_values[
        eruption_window_index
    ]

    title = (
        f"{classifier_name} [s={seed_idx}|idx={eruption_window_index}|p={proba_value:.2f}]\n"
        f"{eruption_window.highest.datetime:%Y-%m-%d %H:%M:%S}"
    )

    return explanation, save_filepath, title


def plot_classifier_waterfall(
    classifier_explanation: ClassifierExplanation,
    classifier_ensemble: ClassifierEnsemble,
    labels: pd.Series | pd.DataFrame,
    eruption_dates: list[str],
    figsize: tuple[float, float] | None = None,
    max_display: int = 20,
    output_dir: str | None = None,
    dpi: int = 150,
    verbose: bool = False,
) -> None:
    """Render the highest-probability waterfall plot per eruption for one classifier.

    Picks the highest-probability window for each eruption from the
    classifier's seed ensemble and renders a SHAP waterfall plot for it.
    Plots land under ``{output_dir}/{eruption_date}/``; the caller is
    responsible for passing an ``eruptions`` root (see
    :meth:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble.plot_waterfall`).

    Args:
        classifier_explanation (ClassifierExplanation): Per-classifier
            SHAP explanations indexed by ``random_state``.
        classifier_ensemble (ClassifierEnsemble): Full ensemble; used to
            pull the matching ``SeedEnsemble`` for
            :func:`~eruption_forecast.utils.ml.build_classifier_ensemble_summary`.
        labels (pd.Series | pd.DataFrame): Ground-truth label series
            indexed by window id.
        eruption_dates (list[str]): Ground-truth eruption dates in
            ``"YYYY-MM-DD"`` format.
        figsize (tuple[float, float] | None, optional): Figure size in
            inches. ``None`` auto-sizes from ``max_display``. Defaults to
            ``None``.
        max_display (int, optional): Maximum number of features to
            display. Defaults to ``20``.
        output_dir (str | None, optional): Root directory for waterfall
            plots — treated as the ``eruptions/`` root. ``None`` falls
            back to ``{cwd}/output/explanation/eruptions``. Defaults to
            ``None``.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to ``150``.
        verbose (bool, optional): Forwarded to :func:`save_figure`.
            Defaults to ``False``.

    Returns:
        None: The function persists its output to disk and has no return
            value.
    """
    eruption_dir = (
        output_dir
        if output_dir is not None
        else os.path.join(os.getcwd(), "output", "explanation", "eruptions")
    )

    classifier_name = classifier_explanation.classifier_name
    seed_ensemble = classifier_ensemble.ensembles[classifier_name]

    classifier_summary = build_classifier_ensemble_summary(
        seed_ensemble=seed_ensemble,
        labels=labels,
        eruption_dates=eruption_dates,
    )

    for eruption_window in classifier_summary.eruption_windows:
        explanation, save_filepath, title = _build_explanation_plot(
            classifier_explanation=classifier_explanation,
            eruption_window=eruption_window,
            eruption_dir=eruption_dir,
            classifier_name=classifier_name,
        )

        plot_shap_waterfall(
            explanation=explanation,
            max_display=max_display,
            title=title,
            figsize=figsize,
            save_filepath=save_filepath,
            dpi=dpi,
            verbose=verbose,
        )

    return None
