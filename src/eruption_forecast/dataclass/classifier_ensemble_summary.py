"""Per-classifier summary of a ``SeedEnsemble``'s probability matrix.

These dataclasses model a per-classifier rollup of per-seed × per-eruption-window
probability picks. Today the summary captures only the highest- and
lowest-probability rows — what a SHAP waterfall renderer needs — but the
schema is not bound to that scope and is free to grow additional aggregates
(mean, std, quantiles, consensus, …) as concrete consumers appear.

The builder lives in :func:`eruption_forecast.utils.ml.build_classifier_ensemble_summary`
and the SHAP waterfall renderer in
:func:`eruption_forecast.plots.explanation_plots.plot_classifier_waterfall`.
This module is intentionally import-light (stdlib + ``pandas`` for the
``pd.Timestamp`` type only) so it stays usable from tests, notebooks, and
prompt-side helpers without pulling matplotlib, shap, or the ensemble
package along.
"""

from dataclasses import field, dataclass

import pandas as pd


@dataclass(frozen=True)
class ProbabilityPick:
    """One (seed, sample) row chosen from the per-seed probability matrix.

    Attributes:
        random_state (int): Seed identifier the pick came from.
        index (int): Row position in the per-seed probability matrix; aligns
            with the SHAP explanation row for the same seed.
        datetime (pd.Timestamp): Timestamp of the sample.
        value (float): Probability value at the picked row.
    """

    random_state: int
    index: int
    datetime: pd.Timestamp
    value: float


@dataclass(frozen=True)
class SeedSummary:
    """Highest- and lowest-probability picks for one seed in one eruption window.

    Attributes:
        random_state (int): Seed identifier shared by both picks.
        highest (ProbabilityPick): Top-probability pick in the window.
        lowest (ProbabilityPick): Bottom-probability pick in the window.
    """

    random_state: int
    highest: ProbabilityPick
    lowest: ProbabilityPick


@dataclass(frozen=True)
class EruptionWindow:
    """All per-seed summaries scoped to a single eruption-date day window.

    Attributes:
        eruption_date (str): Eruption date in ``YYYY-MM-DD`` form.
        highest (ProbabilityPick): Highest-probability pick across all seeds in
            this window. Guaranteed to be set: the builder only constructs an
            :class:`EruptionWindow` when at least one seed contributed.
        lowest (ProbabilityPick): Lowest-probability pick across all seeds in
            this window. Guaranteed to be set under the same construction
            invariant as ``highest``.
        seeds (list[SeedSummary]): One :class:`SeedSummary` per seed.
    """

    eruption_date: str
    highest: ProbabilityPick
    lowest: ProbabilityPick
    seeds: list[SeedSummary]


@dataclass
class ClassifierEnsembleSummary:
    """Per-classifier summary of a ``SeedEnsemble``'s probability matrix.

    One instance is produced per classifier (i.e. per ``SeedEnsemble``); the
    cross-classifier roll-up across a full ``ClassifierEnsemble`` is not
    modelled here and is left to future consumers.

    Attributes:
        classifier_name (str): Classifier this summary covers.
        eruption_windows (list[EruptionWindow]): One :class:`EruptionWindow`
            per eruption date that intersected the prediction grid. Empty
            windows are skipped during construction.
        highest (ProbabilityPick | None): Highest-probability pick across all
            seeds × windows. ``None`` when no eruption window intersected the
            prediction grid.
        lowest (ProbabilityPick | None): Lowest-probability pick across all
            seeds × windows. ``None`` when no eruption window intersected the
            prediction grid.
    """

    classifier_name: str
    eruption_windows: list[EruptionWindow] = field(default_factory=list)
    highest: ProbabilityPick | None = None
    lowest: ProbabilityPick | None = None
