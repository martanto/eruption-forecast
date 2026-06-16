"""Build and render per-classifier SHAP waterfall picks.

Pairs a fitted :class:`~eruption_forecast.ensemble.seed_ensemble.SeedEnsemble`
with the per-seed SHAP explanations produced by
:class:`~eruption_forecast.ensemble.explainer_ensemble.ExplainerEnsemble`,
walks every seed × eruption-date window to locate the most- and
least-confident eruption prediction, and renders the top pick as a SHAP
waterfall chart.

The intermediate data structures live in
:mod:`eruption_forecast.dataclass.waterfall` so they stay importable from
matplotlib- and shap-free contexts (tests, prompts, notebooks).
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.plots.styles import shap_figure
from eruption_forecast.utils.date_utils import to_datetime, set_datetime_index
from eruption_forecast.dataclass.waterfall import (
    SeedWaterfall,
    EruptionWindow,
    WaterfallPoint,
    ClassifierWaterfall,
)
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.ensemble.explainer_ensemble import ClassifierExplanation


def build_classifier_waterfall(
    seed_ensemble: SeedEnsemble,
    labels: pd.Series | pd.DataFrame,
    eruption_dates: list[str],
) -> ClassifierWaterfall:
    """Walk every seed × eruption-window and assemble the waterfall summary.

    For each eruption date the day window ``[00:00, 23:59:59]`` is sliced from
    the per-seed probability matrix cached on
    :attr:`SeedEnsemble.probabilities`. Within that slice each seed contributes
    one :class:`SeedWaterfall` (highest + lowest probability rows). The
    across-windows extrema are tracked in a single pass and exposed on
    :attr:`ClassifierWaterfall.top` / ``.bottom`` so the caller does not have
    to re-scan the structure to find the row to render.

    Args:
        seed_ensemble (SeedEnsemble): Fitted seed ensemble whose
            :attr:`probabilities` attribute has been populated by a prior
            ``predict_with_uncertainty`` / ``save_matrices`` call.
        labels (pd.Series | pd.DataFrame): Datetime-indexed label container
            from the upstream :class:`TrainingModel` / :class:`PredictionModel`
            — used to attach a ``datetime`` column to the probability matrix
            via :func:`set_datetime_index`.
        eruption_dates (list[str]): Eruption dates in ``YYYY-MM-DD`` form.
            Dates that do not intersect the prediction grid are skipped.

    Returns:
        ClassifierWaterfall: Per-classifier waterfall summary.

    Raises:
        RuntimeError: If ``seed_ensemble.probabilities`` is ``None`` — the
            ensemble has not run a prediction yet.
    """
    if seed_ensemble.probabilities is None:
        raise RuntimeError(
            f"{seed_ensemble.classifier_name}: SeedEnsemble has no cached "
            f"probabilities. Run predict_with_uncertainty / forecast first."
        )

    df_probas = set_datetime_index(
        datetime_map=labels.sort_index(),
        df=seed_ensemble.probabilities,
    ).reset_index()

    result = ClassifierWaterfall(classifier_name=seed_ensemble.classifier_name)

    for eruption_date in eruption_dates:
        start_date = to_datetime(eruption_date).replace(hour=0, minute=0, second=0)
        end_date = start_date.replace(hour=23, minute=59, second=59)
        window = df_probas[
            (df_probas["datetime"] >= start_date) & (df_probas["datetime"] <= end_date)
        ]
        if window.empty:
            continue

        seeds_in_window: list[SeedWaterfall] = []
        for seed in seed_ensemble.seeds:
            random_state = int(seed["random_state"])
            column_name = f"seed_{random_state:05d}"

            sorted_window = window.sort_values(column_name, ascending=False)
            top_row = sorted_window.iloc[0]
            bottom_row = sorted_window.iloc[-1]

            seed_waterfall = SeedWaterfall(
                random_state=random_state,
                highest=WaterfallPoint(
                    random_state=random_state,
                    index=int(sorted_window.index[0]),
                    datetime=top_row["datetime"],
                    value=float(top_row[column_name]),
                ),
                lowest=WaterfallPoint(
                    random_state=random_state,
                    index=int(sorted_window.index[-1]),
                    datetime=bottom_row["datetime"],
                    value=float(bottom_row[column_name]),
                ),
            )
            seeds_in_window.append(seed_waterfall)

            if result.top is None or seed_waterfall.highest.value > result.top.value:
                result.top = seed_waterfall.highest
            if (
                result.bottom is None
                or seed_waterfall.lowest.value < result.bottom.value
            ):
                result.bottom = seed_waterfall.lowest

        result.eruption_windows.append(
            EruptionWindow(eruption_date=eruption_date, seeds=seeds_in_window)
        )

    return result


def plot_classifier_waterfall(
    waterfall: ClassifierWaterfall,
    classifier_explanation: ClassifierExplanation,
    max_display: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Render the SHAP waterfall for the top pick on ``waterfall``.

    Looks up the seed identified by :attr:`ClassifierWaterfall.top.random_state`
    inside ``classifier_explanation["seeds"]`` (by ``random_state``, not list
    position), then renders ``shap.plots.waterfall`` for the corresponding
    explanation row.

    Args:
        waterfall (ClassifierWaterfall): Summary returned by
            :func:`build_classifier_waterfall`.
        classifier_explanation (ClassifierExplanation): One of the
            ``TypedDict`` payloads produced by
            :class:`ExplainerEnsemble.explain`. Must contain a ``seeds`` list of
            ``{"random_state": int, "shape_values": shap.Explanation}``.
        max_display (int, optional): Maximum number of features in the
            waterfall. Defaults to 20.
        figsize (tuple[float, float] | None, optional): Figure size in inches.
            When ``None`` uses ``(16.0, max(8.0, max_display * 0.5))`` so long
            tsfresh labels remain readable. Defaults to ``None``.
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: The figure containing the rendered waterfall.

    Raises:
        RuntimeError: If ``waterfall.top`` is ``None`` — no eruption window
            intersected the prediction grid, so there is no row to render.
        KeyError: If the seed identified by ``waterfall.top.random_state`` is
            absent from ``classifier_explanation["seeds"]``.
    """
    if waterfall.top is None:
        raise RuntimeError(
            f"{waterfall.classifier_name}: no eruption windows intersected the "
            f"prediction grid; nothing to render."
        )

    seeds_by_random_state = {
        int(seed["random_state"]): seed for seed in classifier_explanation["seeds"]
    }
    seed = seeds_by_random_state[waterfall.top.random_state]
    explanation: shap.Explanation = seed["shape_values"][waterfall.top.index]

    fig_size = figsize if figsize is not None else (16.0, max(8.0, max_display * 0.5))

    with shap_figure(figsize=fig_size, dpi=dpi) as fig:
        shap.plots.waterfall(
            shap_values=explanation,
            max_display=max_display,
            show=False,
        )
        fig.suptitle(
            f"{waterfall.classifier_name} "
            f"[s={waterfall.top.random_state}|p={waterfall.top.value:.4f}]\n"
            f"{waterfall.top.datetime:%Y-%m-%d %H:%M:%S}",
            fontsize=12,
        )

    return fig
