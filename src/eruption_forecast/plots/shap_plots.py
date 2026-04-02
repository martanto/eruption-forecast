"""SHAP-based model explainability plots for volcanic eruption forecasting.

This module provides publication-quality SHAP visualizations for understanding
model predictions and feature contributions.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import shap
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from shap.maskers import Independent

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir


if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    filepath: str | None = None,
    overwrite: bool = False,
) -> shap.Explanation:
    """Compute a SHAP ``Explanation`` object for the positive class.

    Tree-based ensemble models (detected via ``hasattr(model, "estimators_")``)
    use ``shap.TreeExplainer`` for fast, exact SHAP values. XGBoost models
    (detected via ``hasattr(model, "get_booster")``) and all other classifiers
    (SVM, LR, etc.) use ``shap.Explainer(model.predict_proba, masker)`` with
    an ``Independent`` masker — required for XGBoost ≥ 3.x where
    ``TreeExplainer`` fails with a string-to-float error, and avoids the slow
    ``KernelExplainer`` fallback for non-tree models.

    The positive-class slice is taken automatically for binary classifiers.

    Args:
        model (Any): A fitted scikit-learn compatible estimator with a
            ``predict_proba`` method.
        X (pd.DataFrame): Feature matrix used as both background masker and
            evaluation set.
        feature_names (list[str] | None, optional): Column names used when
            ``X`` is not a DataFrame. Defaults to None.
        filepath (str | None, optional): If provided, the file exists, and
            ``overwrite`` is False, loads and returns the cached
            ``shap.Explanation`` from this path. Otherwise computes fresh
            values and saves them to this path. Defaults to None.
        overwrite (bool, optional): When True, recomputes SHAP values even if
            ``filepath`` already exists. Defaults to False.

    Returns:
        shap.Explanation: SHAP Explanation object with shape
        ``(n_samples, n_features)`` for the positive class.
    """
    if filepath is not None and not overwrite and os.path.isfile(filepath):
        logger.info(f"Loading cached SHAP from {filepath}")
        return joblib.load(filepath)

    cols = list(X.columns) if isinstance(X, pd.DataFrame) else feature_names

    logger.info(
        f"Computing SHAP values for {len(X)} samples and {len(cols)} features..."
    )

    # Tree-based models (RF, GBM, etc.) use the fast exact TreeExplainer.
    # XGBoost ≥ 3.x breaks TreeExplainer with a string-to-float error, so it
    # gets the Independent masker path instead. All other models (SVM, LR,
    # etc.) also use the masker to avoid the slow KernelExplainer fallback.
    if hasattr(model, "estimators_") and not hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model, X)
        explanation: shap.Explanation = explainer(X)
    else:
        masker = Independent(X, max_samples=len(X))
        explainer = shap.Explainer(model.predict_proba, masker)
        explanation: shap.Explanation = explainer(X, silent=True)

    # Binary classifier: shape (n, features, 2) — take positive-class slice.
    if explanation.values.ndim == 3:  # noqa: PD011
        explanation = shap.Explanation(
            values=explanation.values[:, :, 1],  # noqa: PD011
            data=explanation.data,
            feature_names=cols,
        )

    if filepath:
        ensure_dir(os.path.dirname(filepath))
        joblib.dump(explanation, filepath)
        logger.info(
            f"SHAP explanation saved to {filepath}",
        )

    return explanation


def plot_shap_summary(
    model: BaseEstimator,
    X: pd.DataFrame,
    selected_features: list[str] | None = None,
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
    shap_filepath: str | None = None,
) -> plt.Figure:
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
        plt.Figure: Matplotlib figure with the SHAP beeswarm plot.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        >>> fig = plot_shap_summary(rf, X_test)
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
    explanation = compute_shap_explanation(model, X, filepath=shap_filepath)

    fig = plt.figure(figsize=(16, max(8, max_display * 0.5)), dpi=dpi)

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        s=32,  # Default 16
        plot_size=None,
        show=False,
    )

    fig.suptitle(title or "SHAP Summary Plot", y=0.95)

    return fig


def plot_shap_from_file(
    filepath: str,
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, shap.Explanation]:
    """Load a pickled SHAP Explanation and render a beeswarm plot.

    Loads a ``shap.Explanation`` object previously saved with
    ``save_data(..., filetype="pkl")`` via ``joblib.load``, then renders
    a beeswarm plot identical to the one produced by ``plot_shap_summary``.

    Args:
        filepath (str): Path to the ``.pkl`` file containing the saved
            ``shap.Explanation`` object (with or without the extension).
        max_display (int, optional): Maximum number of features to display,
            sorted by mean |SHAP| descending. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            ``"SHAP Summary Plot"``. Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.

    Returns:
        tuple[plt.Figure, shap.Explanation]: Matplotlib figure with the
        beeswarm plot and the loaded ``shap.Explanation`` object.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist (with or without
            the ``.pkl`` extension).

    Examples:
        >>> fig, explanation = plot_shap_from_file("output/shap_values.pkl")
        >>> fig.savefig("shap_summary.png")
    """
    path = filepath if filepath.endswith(".pkl") else f"{filepath}.pkl"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SHAP pickle file not found: {path}")

    explanation: shap.Explanation = joblib.load(path)

    fig = plt.figure(figsize=(16, max(8, max_display * 0.5)), dpi=dpi)
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        s=32,
        plot_size=None,
        show=False,
    )
    fig.suptitle(title or "SHAP Summary Plot", y=0.95)

    return fig, explanation


def _extract_shap_array(shap_values: Any) -> np.ndarray:
    """Extract a 2-D SHAP value array from a SHAP Explanation object or ndarray.

    Handles both raw ``np.ndarray`` returns and ``shap.Explanation`` objects.
    For binary classifiers the SHAP output may have shape ``(n, features, 2)``
    — the positive-class slice ``[:, :, 1]`` is taken automatically.

    Args:
        shap_values (Any): Return value of ``shap.Explainer.__call__``,
            either a ``shap.Explanation`` object or a plain ``np.ndarray``.

    Returns:
        np.ndarray: 2-D array of shape ``(n_samples, n_features)`` containing
        raw SHAP values for the positive class.
    """
    if isinstance(shap_values, np.ndarray):
        raw = shap_values
    elif hasattr(shap_values, "values"):
        raw = np.asarray(shap_values.values)  # noqa: PD011
    else:
        raw = np.array(shap_values)

    # Binary classifier: (n, features, 2) → class-1 slice
    if raw.ndim == 3:
        raw = raw[:, :, 1]
    return raw


def _build_aggregate_explanation(
    explanations: list[shap.Explanation | None],
    per_seed_names: list[list[str]],
    all_names: list[str],
) -> shap.Explanation:
    """Build a merged SHAP Explanation by zero-padding seeds to the union feature space.

    Each non-None explanation is zero-padded from its per-seed feature set to
    the full union feature space, then all valid seeds are concatenated along
    the sample axis into a single ``shap.Explanation`` object ready for
    ``shap.plots.beeswarm``.

    Args:
        explanations (list[shap.Explanation | None]): Per-seed Explanation
            objects; ``None`` entries are skipped.
        per_seed_names (list[list[str]]): Feature name lists, one per seed,
            aligned with ``explanations``.
        all_names (list[str]): Ordered union of all feature names (the target
            feature space for zero-padding).

    Returns:
        shap.Explanation: Merged Explanation with shape
        ``(total_samples, n_union_features)`` and ``feature_names=all_names``.

    Raises:
        ValueError: If no valid (non-None) explanations are provided.
    """
    n_features = len(all_names)
    feat_idx: dict[str, int] = {name: i for i, name in enumerate(all_names)}

    all_values: list[np.ndarray] = []
    all_data: list[np.ndarray] = []

    for exp, seed_names in zip(explanations, per_seed_names, strict=True):
        if exp is None:
            continue
        vals = np.asarray(exp.values)  # noqa: PD011
        raw_data = getattr(exp, "data", None)
        data = np.asarray(raw_data) if raw_data is not None else np.zeros_like(vals)
        n_samples = vals.shape[0]

        padded_vals = np.zeros((n_samples, n_features))
        padded_data = np.zeros((n_samples, n_features))
        for j, name in enumerate(seed_names):
            if name in feat_idx:
                padded_vals[:, feat_idx[name]] = vals[:, j]
                padded_data[:, feat_idx[name]] = data[:, j]

        all_values.append(padded_vals)
        all_data.append(padded_data)

    if not all_values:
        raise ValueError("No valid seeds to build aggregate explanation.")

    merged_values = np.concatenate(all_values, axis=0)
    merged_data = np.concatenate(all_data, axis=0)
    return shap.Explanation(
        values=merged_values,
        data=merged_data,
        feature_names=all_names,
    )


def plot_aggregate_shap_summary(
    models: list[BaseEstimator],
    X_tests: list[pd.DataFrame],
    feature_names: list[list[str]] | list[str],
    max_display: int = 20,
    title: str | None = None,
    dpi: int = 150,
    explanations: list[shap.Explanation | None] | None = None,
) -> tuple[plt.Figure, shap.Explanation]:
    """Plot a beeswarm of SHAP values aggregated across multiple seeds.

    Gathers raw SHAP values from all seeds, zero-pads each seed's values to
    the union feature space, concatenates them, and renders a single
    ``shap.plots.beeswarm`` that shows both magnitude and direction of feature
    contributions. Seeds whose SHAP computation fails are skipped with a warning.

    Args:
        models (list[BaseEstimator]): List of fitted estimators, one per seed.
        X_tests (list[pd.DataFrame]): Corresponding test feature DataFrames,
            one per seed.
        feature_names (list[list[str]] | list[str]): Either a flat list of
            feature names shared by all seeds, or a list of per-seed feature
            name lists.
        max_display (int, optional): Maximum number of features to display,
            sorted by mean |SHAP| descending. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            "Aggregate SHAP Beeswarm". Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults
            to 150.
        explanations (list[shap.Explanation | None] | None, optional):
            Pre-computed SHAP Explanation objects, one per seed. When a slot
            is not None the cached values are used directly. Defaults to None.

    Returns:
        tuple[plt.Figure, shap.Explanation]: Matplotlib figure with the
        beeswarm plot and the merged ``shap.Explanation`` used to produce it.

    Raises:
        ValueError: If ``models``, ``X_tests``, and ``feature_names`` differ
            in length, or if no seeds produced valid SHAP values.

    Examples:
        >>> fig, agg_exp = plot_aggregate_shap_summary(
        ...     models=trained_models,
        ...     X_tests=test_sets,
        ...     feature_names=per_seed_feature_names,
        ...     max_display=15,
        ... )
        >>> fig.savefig("aggregate_shap_beeswarm.png")
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

    # Compute or collect per-seed explanations
    seed_explanations: list[shap.Explanation | None] = []
    for i, (model, X) in enumerate(zip(models, X_tests, strict=True)):
        cached = explanations[i] if explanations is not None else None
        if cached is not None:
            seed_explanations.append(cached)
        else:
            try:
                seed_explanations.append(compute_shap_explanation(model, X))
            except Exception as e:
                logger.warning(f"Skipping seed {i} in SHAP beeswarm aggregation: {e}")
                seed_explanations.append(None)

    agg_explanation = _build_aggregate_explanation(
        seed_explanations, per_seed_names, all_names
    )

    fig = plt.figure(figsize=(16, max(8, max_display * 0.5)), dpi=dpi)
    shap.plots.beeswarm(
        agg_explanation,
        max_display=max_display,
        s=32,
        plot_size=None,
        show=False,
    )
    fig.suptitle(title or "Aggregate SHAP Beeswarm", y=0.95)

    return fig, agg_explanation
