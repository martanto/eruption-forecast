"""SHAP-based model explainability plots for volcanic eruption forecasting.

This module provides publication-quality SHAP visualizations for understanding
model predictions and feature contributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    configure_spine,
    apply_nature_style,
)


if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def plot_shap_summary(
    model: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot a SHAP beeswarm summary for a single fitted model.

    Uses ``shap.Explainer`` to auto-select the best explainer (TreeExplainer
    for tree models, LinearExplainer for linear models, etc.), then renders
    a beeswarm dot plot showing both direction and magnitude of feature
    contributions for the top ``max_display`` features.

    When ``X`` is a plain ``np.ndarray`` and ``feature_names`` is provided,
    ``X`` is wrapped in a ``pd.DataFrame`` so that SHAP axis labels are
    populated correctly.

    Args:
        model (BaseEstimator): Fitted scikit-learn estimator to explain.
        X (np.ndarray | pd.DataFrame): Feature matrix used to compute SHAP
            values. Should be the test set or a representative sample.
        feature_names (list[str] | None, optional): Feature names for the
            axis labels. Applied when ``X`` is a plain ``np.ndarray``; ignored
            when ``X`` is already a ``DataFrame`` (column names are used
            directly). Defaults to None.
        max_display (int, optional): Maximum number of features to display,
            sorted by mean |SHAP| descending. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            "SHAP Summary Plot". Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.

    Returns:
        plt.Figure: Matplotlib figure with the SHAP beeswarm plot.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        >>> fig = plot_shap_summary(rf, X_test, feature_names=X_test.columns.tolist())
        >>> fig.savefig("shap_summary.png")
    """
    # Wrap ndarray with feature names so SHAP labels axes correctly.
    if feature_names is not None and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    # Do not pass X as background masker — shap.Explainer(model, X) fails when
    # matplotlib rcParams have been customised (shap 0.49 bug). TreeExplainer
    # does not require background data; X is still used as the evaluation set.
    explainer = shap.Explainer(model)
    try:
        shap_values = explainer(X)
    except Exception as e:
        logger.warning(f"SHAP summary plot: {e}")
        shap_values = explainer(X, check_additivity=False)

    # For binary classifiers shap_values may have shape (n, features, 2);
    # take the positive-class slice via Explanation indexing.
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:  # noqa: PD011
        shap_values = shap_values[:, :, 1]

    with apply_nature_style():
        # shap.plots.beeswarm always draws on the current figure; capture it
        # immediately after the call rather than diffing figure numbers, which
        # is fragile when other threads or callbacks create figures concurrently.
        shap.plots.beeswarm(
            shap_values,
            max_display=max_display,
            s=32,  # Default 16
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(20, max(8, max_display * 0.5))
        fig.set_dpi(dpi)
        fig.suptitle(title or "SHAP Summary Plot", y=1.02)

    return fig


def _extract_shap_array(shap_output: Any) -> np.ndarray:
    """Extract a 2-D SHAP value array from a SHAP Explanation object or ndarray.

    Handles both raw ``np.ndarray`` returns and ``shap.Explanation`` objects.
    For binary classifiers the SHAP output may have shape ``(n, features, 2)``
    — the positive-class slice ``[:, :, 1]`` is taken automatically.

    Args:
        shap_output (Any): Return value of ``shap.Explainer.__call__``,
            either a ``shap.Explanation`` object or a plain ``np.ndarray``.

    Returns:
        np.ndarray: 2-D array of shape ``(n_samples, n_features)`` containing
        raw SHAP values for the positive class.
    """
    if isinstance(shap_output, np.ndarray):
        raw = shap_output
    elif hasattr(shap_output, "values"):
        raw = np.asarray(shap_output.values)  # noqa: PD011
    else:
        raw = np.array(shap_output)

    # Binary classifier: (n, features, 2) → class-1 slice
    if raw.ndim == 3:
        raw = raw[:, :, 1]
    return raw


def plot_aggregate_shap_summary(
    models: list[BaseEstimator],
    X_tests: list[pd.DataFrame],
    feature_names: list[list[str]] | list[str],
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot mean absolute SHAP values aggregated across multiple seeds.

    Computes per-seed mean |SHAP| vectors, aligns them to the union of all
    seed feature sets (filling absent features with zero), then plots the
    mean ± std across seeds as a horizontal bar chart. Seeds whose SHAP
    computation fails are skipped with a warning.

    When all seeds share an identical feature set (``feature_names`` is a
    flat ``list[str]``), the behaviour is equivalent to the original
    implementation. When seeds use different significant-feature subsets
    (``feature_names`` is a ``list[list[str]]``), missing features are
    treated as zero importance for that seed.

    Args:
        models (list[BaseEstimator]): List of fitted estimators, one per
            random seed.
        X_tests (list[pd.DataFrame]): Corresponding test feature DataFrames,
            one per seed. Each DataFrame's columns must match the
            corresponding entry in ``feature_names``.
        feature_names (list[list[str]] | list[str]): Either a flat list of
            feature names shared by all seeds, or a list of per-seed feature
            name lists. Lengths must match ``models`` and ``X_tests`` when
            nested.
        max_display (int, optional): Number of top features to show in
            the bar chart, sorted by mean |SHAP| descending. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            "Aggregate SHAP Feature Importance". Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``feature``, ``mean_shap``, ``std_shap`` sorted
            by descending ``mean_shap`` (all features, not just top-N).

    Raises:
        ValueError: If ``models``, ``X_tests``, and ``feature_names`` differ
            in length, or if no seeds produced valid SHAP values.

    Examples:
        >>> fig, df = plot_aggregate_shap_summary(
        ...     models=trained_models,
        ...     X_tests=test_sets,
        ...     feature_names=per_seed_feature_names,
        ...     max_display=15,
        ... )
        >>> fig.savefig("aggregate_shap.png")
        >>> df.head()
    """
    # Normalise feature_names to per-seed lists
    per_seed_names: list[list[str]]
    if feature_names and isinstance(feature_names[0], list):
        per_seed_names = feature_names  # type: ignore[assignment]
    else:
        per_seed_names = [feature_names] * len(models)  # type: ignore[list-item]

    if not (len(models) == len(X_tests) == len(per_seed_names)):
        raise ValueError(
            f"models ({len(models)}), X_tests ({len(X_tests)}), and "
            f"feature_names ({len(per_seed_names)}) must have equal length."
        )

    # Build union feature space (insertion-ordered)
    all_names: list[str] = list(
        dict.fromkeys(name for names in per_seed_names for name in names)
    )
    n_features = len(all_names)
    feat_idx: dict[str, int] = {name: i for i, name in enumerate(all_names)}

    all_abs_shap: list[np.ndarray] = []

    for model, X, seed_names in zip(models, X_tests, per_seed_names, strict=True):
        try:
            # Do not pass X as background masker — see note in plot_shap_summary.
            explainer = shap.Explainer(model)
            try:
                shap_output = explainer(X)
            except Exception as e:
                logger.warning(f"SHAP aggregate summary plot: {e}")
                shap_output = explainer(X, check_additivity=False)

            vals = _extract_shap_array(shap_output)
            seed_abs_mean = np.abs(vals).mean(axis=0)  # (n_seed_features,)

            # Map seed values into the union feature space
            row = np.zeros(n_features)
            for j, name in enumerate(seed_names):
                if name in feat_idx:
                    row[feat_idx[name]] = seed_abs_mean[j]

            all_abs_shap.append(row)
        except Exception as e:
            logger.warning(f"Skipping seed in SHAP aggregation: {e}")

    if not all_abs_shap:
        raise ValueError("No seeds produced valid SHAP values.")

    stacked = np.stack(all_abs_shap, axis=0)  # (n_valid_seeds, n_features)
    mean_shap = stacked.mean(axis=0)
    std_shap = stacked.std(axis=0)

    # Sort by mean descending
    sorted_idx = np.argsort(mean_shap)[::-1]
    top_idx = sorted_idx[:max_display]

    # Reverse so the highest-ranked feature appears at the top of the chart
    # (matplotlib barh plots from bottom to top by default).
    display_idx = top_idx[::-1]
    display_features = [all_names[i] for i in display_idx]
    display_mean = mean_shap[display_idx]
    display_std = std_shap[display_idx]

    figheight = max(4, max_display * 0.35)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(8, figheight), dpi=dpi)

        ax.barh(
            range(len(display_features)),
            display_mean,
            xerr=display_std,
            color=OKABE_ITO[4],
            alpha=0.8,
            error_kw={"ecolor": "gray", "capsize": 3, "linewidth": 1.0},
        )
        ax.set_yticks(range(len(display_features)))
        ax.set_yticklabels(display_features)

        configure_spine(ax)
        ax.set_xlabel("Mean |SHAP value| ± Std")
        ax.set_ylabel("Feature")
        ax.set_title(title or "Aggregate SHAP Feature Importance")

    # Return all features sorted by descending mean
    data = pd.DataFrame(
        {
            "feature": [all_names[i] for i in sorted_idx],
            "mean_shap": mean_shap[sorted_idx],
            "std_shap": std_shap[sorted_idx],
        }
    )
    return fig, data
