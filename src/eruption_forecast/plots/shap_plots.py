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
    X: pd.DataFrame,
    selected_features: list[str] | None = None,
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, shap.Explanation]:
    """Plot a SHAP beeswarm summary for a single fitted model.

    Renders a beeswarm dot plot showing both direction and magnitude of
    feature contributions for the top ``max_display`` features. Feature names
    are read directly from ``X.columns``.

    Args:
        model (BaseEstimator): Fitted scikit-learn estimator to explain.
        X (pd.DataFrame): Feature matrix used to compute SHAP values.
            Should be the test set or a representative sample.
        selected_features (list[str] | None, optional): If provided, restrict
            SHAP computation to these columns only. Defaults to None.
        max_display (int, optional): Maximum number of features to display,
            sorted by mean |SHAP| descending. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            "SHAP Summary Plot". Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.

    Returns:
        tuple[plt.Figure, shap.Explanation]: Matplotlib figure with the SHAP
        beeswarm plot and the SHAP Explanation object used to produce it.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        >>> fig, explanation = plot_shap_summary(rf, X_test)
        >>> fig.savefig("shap_summary.png")
    """
    if selected_features is not None:
        X = X[selected_features]

    # Use predict_proba + Independent masker so shap.Explainer works for any
    # classifier, including XGBClassifier with XGBoost ≥ 3.x where
    # shap.TreeExplainer fails with a string-to-float conversion error.
    # The masker is built from X itself (all rows as background).
    # shap.Explainer(model) without a masker raises "not callable" for some
    # model types, so the masker is always passed explicitly here.
    shap_explanation = _compute_shap_explanation(model, X)

    with apply_nature_style():
        # shap.plots.beeswarm always draws on the current figure; capture it
        # immediately after the call rather than diffing figure numbers, which
        # is fragile when other threads or callbacks create figures concurrently.
        shap.plots.beeswarm(
            shap_explanation,
            max_display=max_display,
            s=32,  # Default 16
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(20, max(8, max_display * 0.5))
        fig.set_dpi(dpi)
        fig.suptitle(title or "SHAP Summary Plot", y=1.02)

    return fig, shap_explanation


def _compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute a SHAP ``Explanation`` object for the positive class.

    Uses ``shap.Explainer(model.predict_proba, masker)`` with an
    ``Independent`` masker built from ``X``. This path works for any
    scikit-learn compatible classifier, including ``XGBClassifier`` with
    XGBoost ≥ 3.x where ``shap.TreeExplainer`` fails with a string-to-float
    conversion error, and avoids the "not callable" error raised by
    ``shap.Explainer(model)`` when no masker is supplied.

    For binary classifiers the explainer returns shape ``(n, features, 2)``;
    the positive-class slice ``[:, :, 1]`` is taken automatically.

    Args:
        model (Any): A fitted scikit-learn compatible estimator with a
            ``predict_proba`` method.
        X (pd.DataFrame): Feature matrix used as both background masker and
            evaluation set.
        feature_names (list[str] | None, optional): Column names used when
            ``X`` is not a DataFrame. Defaults to None.

    Returns:
        shap.Explanation: SHAP Explanation object with shape
        ``(n_samples, n_features)`` for the positive class.
    """
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else feature_names
    masker = shap.maskers.Independent(X, max_samples=len(X))
    explainer = shap.Explainer(model.predict_proba, masker)
    shap_output = explainer(X)
    # Binary classifier: shape (n, features, 2) — take positive-class slice.
    if shap_output.values.ndim == 3:  # noqa: PD011
        return shap.Explanation(
            values=shap_output.values[:, :, 1],  # noqa: PD011
            data=shap_output.data,
            feature_names=cols,
        )
    return shap_output


def _compute_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Compute a 2-D SHAP value array for the positive class.

    Delegates to ``_compute_shap_explanation`` and extracts the raw values
    array. Returns a plain ``np.ndarray`` of shape ``(n_samples, n_features)``
    suitable for numerical aggregation.

    Args:
        model (Any): A fitted scikit-learn compatible estimator.
        X (pd.DataFrame): Feature matrix used to compute SHAP values.

    Returns:
        np.ndarray: 2-D array of shape ``(n_samples, n_features)`` containing
        SHAP values for the positive class.
    """
    explanation = _compute_shap_explanation(model, X)
    return np.asarray(explanation.values)  # noqa: PD011


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
    explanations: list[shap.Explanation | None] | None = None,
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
        explanations (list[shap.Explanation | None] | None, optional):
            Pre-computed SHAP Explanation objects, one per seed. When a
            slot is not None the cached values are used directly instead of
            recomputing. Defaults to None (all seeds are computed on the fly).

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

    for i, (model, X, seed_names) in enumerate(
        zip(models, X_tests, per_seed_names, strict=True)
    ):
        cached = explanations[i] if explanations is not None else None
        try:
            if cached is not None:
                vals = np.asarray(cached.values)  # noqa: PD011
            else:
                vals = _compute_shap_values(model, X)
            seed_abs_mean = np.abs(vals).mean(axis=0)  # (n_seed_features,)

            # Map seed values into the union feature space
            row = np.zeros(n_features)
            for j, name in enumerate(seed_names):
                if name in feat_idx:
                    row[feat_idx[name]] = seed_abs_mean[j]

            all_abs_shap.append(row)
        except Exception as e:
            logger.warning(f"Skipping seed {i} in SHAP aggregation: {e}")

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
