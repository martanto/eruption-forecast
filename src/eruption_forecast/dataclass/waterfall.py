"""Pure data structures describing a per-classifier SHAP waterfall pick.

These dataclasses model the result of walking a ``SeedEnsemble``'s per-seed
probability matrix to find — for each seed within each eruption-date window —
the highest- and lowest-probability sample. The actual walk + plotting live in
:mod:`eruption_forecast.plots.shap_waterfall`; this module is intentionally
import-light so it stays usable from tests, notebooks, and prompt-side helpers
without pulling matplotlib, shap, or the ensemble package along.
"""

from dataclasses import field, dataclass

import pandas as pd


@dataclass(frozen=True)
class WaterfallPoint:
    """One (seed, sample) pick — either the highest or lowest probability row.

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
class SeedWaterfall:
    """Highest / lowest probability rows for one seed in one eruption window.

    Attributes:
        random_state (int): Seed identifier shared by both picks.
        highest (WaterfallPoint): Top-probability pick in the window.
        lowest (WaterfallPoint): Bottom-probability pick in the window.
    """

    random_state: int
    highest: WaterfallPoint
    lowest: WaterfallPoint


@dataclass(frozen=True)
class EruptionWindow:
    """All per-seed picks scoped to a single eruption-date day window.

    Attributes:
        eruption_date (str): Eruption date in ``YYYY-MM-DD`` form.
        seeds (list[SeedWaterfall]): One :class:`SeedWaterfall` per seed.
    """

    eruption_date: str
    seeds: list[SeedWaterfall]


@dataclass
class ClassifierWaterfall:
    """All eruption windows + across-windows top/bottom for one classifier.

    Attributes:
        classifier_name (str): Classifier this waterfall summarises.
        eruption_windows (list[EruptionWindow]): One :class:`EruptionWindow`
            per eruption date that intersected the prediction grid. Empty
            windows are skipped during construction.
        top (WaterfallPoint | None): Highest-probability pick across all
            seeds × windows; the row to render in a waterfall plot. ``None``
            when no window intersected the prediction grid.
        bottom (WaterfallPoint | None): Lowest-probability pick across all
            seeds × windows. ``None`` when no window intersected the
            prediction grid.
    """

    classifier_name: str
    eruption_windows: list[EruptionWindow] = field(default_factory=list)
    top: WaterfallPoint | None = None
    bottom: WaterfallPoint | None = None
