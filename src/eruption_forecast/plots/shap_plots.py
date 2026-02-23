"""SHAP-based model explainability plots for volcanic eruption forecasting.

This module provides publication-quality SHAP visualizations for understanding
model predictions and feature contributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Args:
        model (BaseEstimator): Fitted scikit-learn estimator to explain.
        X (np.ndarray | pd.DataFrame): Feature matrix used to compute SHAP
            values. Should be the test set or a representative sample.
        feature_names (list[str] | None, optional): Feature names for the
            x-axis labels. If None and X is a DataFrame, column names are
            used automatically. Defaults to None.
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
    explainer = shap.Explainer(model, X)
    try:
        shap_values = explainer(X)
    except Exception as e:
        logger.warning(f"SHAP summary plot: {e}")
        shap_values = explainer(X, check_additivity=False)

    # For binary classifiers shap_values may have shape (n, features, 2);
    # take the positive-class slice.
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:  # noqa: PD011
        shap_values = shap_values[:, :, 1]

    with apply_nature_style():
        existing_fignums = set(plt.get_fignums())
        shap.plots.beeswarm(
            shap_values,
            max_display=max_display,
            s=32,  # Default 16
            show=False,
        )
        new_fignums = set(plt.get_fignums()) - existing_fignums
        fig = plt.figure(new_fignums.pop()) if new_fignums else plt.gcf()
        fig.set_size_inches(20, max(8, max_display * 0.5))
        fig.set_dpi(dpi)
        fig.suptitle(title or "SHAP Summary Plot", y=1.02)

    return fig


def plot_aggregate_shap_summary(
    models: list[BaseEstimator],
    X_tests: list[np.ndarray | pd.DataFrame],
    feature_names: list[str],
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot mean absolute SHAP values aggregated across multiple seeds.

    Concatenates SHAP values from all seeds and plots mean |SHAP| per
    feature as a horizontal bar chart with ±1 std error bars. This gives
    an ensemble-level view of feature importance with direction-agnostic
    magnitude.

    Args:
        models (list[BaseEstimator]): List of fitted estimators, one per
            random seed.
        X_tests (list[np.ndarray | pd.DataFrame]): Corresponding test
            feature matrices, one per seed.
        feature_names (list[str]): Feature names matching the columns in
            each X_test.
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

    Examples:
        >>> fig, df = plot_aggregate_shap_summary(
        ...     models=trained_models,
        ...     X_tests=test_sets,
        ...     feature_names=feature_names,
        ...     max_display=15,
        ... )
        >>> fig.savefig("aggregate_shap.png")
        >>> df.head()
    """
    all_abs_shap: list[np.ndarray] = []

    for model, X in zip(models, X_tests, strict=False):
        explainer = shap.Explainer(model, X)
        try:
            shap_values = explainer(X)
        except Exception as e:
            logger.warning(f"SHAP aggregate summary plot: {e}")
            shap_values = explainer(X, check_additivity=False)

        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:  # noqa: PD011
            vals = shap_values.values[:, :, 1]  # noqa: PD011
        else:
            vals = shap_values.values  # noqa: PD011

        all_abs_shap.append(np.abs(vals).mean(axis=0))

    stacked = np.stack(all_abs_shap, axis=0)  # (n_seeds, n_features)
    mean_shap = stacked.mean(axis=0)
    std_shap = stacked.std(axis=0)

    # Sort by mean descending
    sorted_idx = np.argsort(mean_shap)[::-1]
    top_idx = sorted_idx[:max_display]

    # Reverse for bottom-to-top bar display
    display_idx = top_idx[::-1]
    display_features = [feature_names[i] for i in display_idx]
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
            "feature": [feature_names[i] for i in sorted_idx],
            "mean_shap": mean_shap[sorted_idx],
            "std_shap": std_shap[sorted_idx],
        }
    )
    return fig, data
