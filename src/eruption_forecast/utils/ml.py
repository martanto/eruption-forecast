import os
import json
from typing import Any, Literal
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)
from tsfresh.transformers import FeatureSelector
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV

from eruption_forecast.logger import logger
from eruption_forecast.utils.dataframe import to_series
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.utils.date_utils import (
    sort_dates,
    to_datetime,
    to_datetime_index,
)
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.dataclass.classifier_ensemble_summary import (
    SeedSummary,
    EruptionWindow,
    ProbabilityPick,
    ClassifierEnsembleSummary,
)


def build_y_true(
    features_label_csv: str,
    eruption_dates: list[str] | list[datetime],
    datetime_column: str = "datetime",
) -> pd.Series:
    """Build a pandas Series containing ground truth binary labels (y_true).

    Reads a features-label CSV indexed by datetime, marks every row falling on
    one of ``eruption_dates`` as ``1`` in the ``is_erupted`` column, and returns
    the column re-indexed by the window ``id``. Rows outside eruption days are
    coerced to ``0`` so the returned series never contains ``NaN``.

    Args:
        features_label_csv (str): Path to the features-label CSV. Usually saved
            under ``output_dir/{prediction,training}/features/features-label*.csv``.
            The CSV must contain ``id`` and ``is_erupted`` columns plus the
            datetime column named by ``datetime``.
        eruption_dates (list[str] | list[datetime]): Eruption dates. Strings are
            parsed via :func:`sort_dates`.
        datetime_column (str, optional): Name of the column to use as the
            datetime index. Forwarded to ``pd.read_csv(index_col=...)``.
            Defaults to ``"datetime"``.

    Returns:
        pd.Series: Ground truth binary labels indexed by window ``id``, with
            name ``"is_erupted"``.

    Raises:
        FileNotFoundError: If ``features_label_csv`` does not exist.
        KeyError: If the CSV is missing the ``id`` or ``is_erupted`` column,
            or if ``datetime_column`` is not present in the CSV.
        TypeError: If the resolved index cannot be parsed as a
            ``pd.DatetimeIndex``.

    Examples:
        >>> y_true = build_y_true(
        ...     "output/VG.OJN.00.EHZ/prediction/features/features-label.csv",
        ...     eruption_dates=["2025-03-20"],
        ... )
        >>> y_true.value_counts()
    """
    df = pd.read_csv(features_label_csv, index_col=datetime_column, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"features_label_csv index must be a pd.DatetimeIndex, "
            f"got '{type(df.index).__name__}'. "
            f"Check that column '{datetime_column}' in '{features_label_csv}' "
            f"contains parseable datetimes."
        )

    missing = {"id", "is_erupted"} - set(df.columns)
    if missing:
        raise KeyError(
            f"features_label_csv is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}."
        )

    df["is_erupted"] = df["is_erupted"].fillna(0).astype(int)

    sorted_eruption_dates = sort_dates(eruption_dates, as_datetime=True)
    for eruption_date in sorted_eruption_dates:
        start_date = eruption_date.replace(hour=0, minute=0, second=0)
        end_date = eruption_date.replace(hour=23, minute=59, second=59)
        df.loc[start_date:end_date, "is_erupted"] = 1

    y_true = df.set_index("id")["is_erupted"]
    y_true.name = "is_erupted"

    return y_true


def temporal_train_test_split(
    eruption_dates: list[str] | list[datetime],
    test_size: float = 0.2,
) -> tuple[list[str] | list[datetime], list[str] | list[datetime]]:
    """Split eruption dates into train and test sets while preserving order.

    Time-series analogue of :func:`sklearn.model_selection.train_test_split`.
    Sorts ``eruption_dates`` chronologically and assigns the most recent
    ``test_size`` fraction to the test split — never shuffles, so the train
    split is strictly older than the test split and no future leakage occurs.
    The split index is clamped to ``[1, n - 1]`` so neither side is ever empty.

    Args:
        eruption_dates (list[str] | list[datetime]): Eruption dates in any
            order. Mixed-type lists are sorted via :func:`sort_dates`.
        test_size (float, optional): Fraction of dates assigned to the test
            split (most recent dates). Must lie in ``[0, 1]``. Defaults to
            ``0.2``.

    Returns:
        tuple[list, list]: ``(train_dates, test_dates)``. ``train_dates``
            contains the older dates; ``test_dates`` contains the most recent
            dates. Both lists are sorted chronologically and preserve the
            input element type.

    Raises:
        ValueError: If ``test_size`` is outside ``[0, 1]``.

    Examples:
        >>> train, test = temporal_train_test_split(
        ...     ["2024-06-15", "2025-03-20", "2024-12-01", "2025-08-10"],
        ...     test_size=0.25,
        ... )
        >>> train
        ['2024-06-15', '2024-12-01', '2025-03-20']
        >>> test
        ['2025-08-10']
    """
    if test_size < 0 or test_size > 1:
        raise ValueError(f"test_size must be between 0 and 1. You provided {test_size}")

    eruption_dates = sort_dates(eruption_dates, as_datetime=False)

    # Clamp to ``[1, n - 1]`` so the call always yields a non-empty train
    # split AND a non-empty test split, even when ``test_size`` rounds to 0
    # or n on small inputs.
    split_idx = np.clip(
        np.ceil(len(eruption_dates) * test_size),
        a_min=1,
        a_max=len(eruption_dates) - 1,
    ).astype(int)

    train_dates = eruption_dates[:-split_idx]
    test_dates = eruption_dates[-split_idx:]
    return train_dates, test_dates


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    resolution: int = 101,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Sweep decision thresholds and compute classification metrics at each step.

    Iterates over ``resolution`` evenly-spaced thresholds from 0.0 to 1.0,
    binarises ``y_proba`` at each step, and records precision, recall, F1,
    balanced accuracy, and MCC. This is the single source of truth for threshold
    analysis used by both :class:`MetricsEnsemble` and ``plot_threshold_analysis``.

    Args:
        y_true (np.ndarray): Ground-truth binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        resolution (int, optional): Number of threshold steps. Defaults to ``101``.

    Returns:
        tuple[np.ndarray, dict[str, list[float]]]: A 2-tuple of:
            - thresholds: 1-D array of length ``resolution`` from 0.0 to 1.0.
            - metrics_dict: dict with keys ``"precision"``, ``"recall"``,
              ``"f1"``, ``"balanced_accuracy"``, ``"specificity"``,
              ``"mcc"``, and ``"g_mean"``, each a list of floats.
    """
    thresholds = np.linspace(0.0, 1.0, resolution)
    metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "balanced_accuracy": [],
        "specificity": [],
        "mcc": [],
        "g_mean": [],
    }

    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
        tn, fp, _, _ = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics["f1"].append(f1_score(y_true, y_pred_thresh, zero_division=0))
        metrics["recall"].append(recall)
        metrics["precision"].append(
            precision_score(y_true, y_pred_thresh, zero_division=0)
        )
        metrics["balanced_accuracy"].append(
            balanced_accuracy_score(y_true, y_pred_thresh)
        )
        metrics["specificity"].append(specificity)
        metrics["mcc"].append(matthews_corrcoef(y_true, y_pred_thresh))
        metrics["g_mean"].append(float(np.sqrt(recall * specificity)))

    return thresholds, metrics


def compute_aggregate_threshold_metrics(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    resolution: int = 101,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Aggregate per-seed threshold metrics into mean ± std curves.

    For every ``(y_true, y_proba)`` pair, calls
    :func:`compute_threshold_metrics`, stacks the resulting per-seed curves
    along a new seed axis, and returns the mean and standard deviation at each
    threshold step plus the raw per-seed stack for callers that need to plot
    individual curves.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): Ground-truth binary labels.
            A single ``np.ndarray`` is broadcast to every seed; a list must
            have the same length as ``y_probas``.
        y_probas (list[np.ndarray]): Per-seed predicted probabilities for the
            positive class. Must contain at least one seed.
        resolution (int, optional): Number of threshold steps forwarded to
            :func:`compute_threshold_metrics`. Defaults to ``101``.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
            A 4-tuple of:

            - ``thresholds``: 1-D array of shape ``(resolution,)``.
            - ``mean_curves``: dict keyed by metric name; each value has
              shape ``(resolution,)``.
            - ``std_curves``: dict keyed by metric name; each value has
              shape ``(resolution,)``.
            - ``per_seed_curves``: dict keyed by metric name; each value has
              shape ``(n_seeds, resolution)``.

            Metric keys come from :func:`compute_threshold_metrics`; keys
            whose lists were never populated are skipped (a defensive
            guard — every metric key is currently populated on every call).

    Raises:
        ValueError: If ``y_probas`` is empty, or if ``y_trues`` is a list whose
            length does not match ``y_probas``.
    """
    if len(y_probas) == 0:
        raise ValueError("y_probas must contain at least one seed.")

    if isinstance(y_trues, np.ndarray):
        y_trues_list: list[np.ndarray] = [y_trues] * len(y_probas)
    else:
        y_trues_list = list(y_trues)

    if len(y_trues_list) != len(y_probas):
        raise ValueError(
            f"y_trues length ({len(y_trues_list)}) does not match "
            f"y_probas length ({len(y_probas)})."
        )

    all_curves: dict[str, list[np.ndarray]] = {}
    thresholds: np.ndarray | None = None

    for y_true, y_proba in zip(y_trues_list, y_probas, strict=True):
        thresholds, metrics = compute_threshold_metrics(
            y_true, y_proba, resolution=resolution
        )
        for key, values in metrics.items():
            if not values:
                continue
            all_curves.setdefault(key, []).append(np.array(values))

    assert thresholds is not None  # guaranteed by the non-empty y_probas check

    per_seed_curves: dict[str, np.ndarray] = {
        key: np.stack(curves, axis=0) for key, curves in all_curves.items()
    }
    mean_curves: dict[str, np.ndarray] = {
        key: matrix.mean(axis=0) for key, matrix in per_seed_curves.items()
    }
    std_curves: dict[str, np.ndarray] = {
        key: matrix.std(axis=0) for key, matrix in per_seed_curves.items()
    }

    return thresholds, mean_curves, std_curves, per_seed_curves


def random_under_sampler(
    features: pd.DataFrame,
    labels: pd.Series,
    sampling_strategy: str | float = "auto",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply random under-sampling to balance classes.

    Handles imbalanced eruption/non-eruption datasets by randomly removing
    samples from the majority class (non-eruption) to match the minority
    class (eruption) based on the sampling strategy. This improves classifier
    performance on imbalanced data.

    Args:
        features (pd.DataFrame): Features DataFrame with training samples.
        labels (pd.Series): Binary labels Series (0=non-eruption, 1=eruption).
        sampling_strategy (str | float, optional): Sampling ratio or strategy.
            If "auto", balances to 50/50. If float, represents desired ratio
            of minority/majority class. Defaults to "auto".
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - features (pd.DataFrame): Balanced features DataFrame.
            - labels (pd.Series): Balanced labels Series.

    Examples:
        >>> balanced_X, balanced_y = random_under_sampler(
        ...     features, labels, sampling_strategy=0.75, random_state=42
        ... )
        >>> print(balanced_y.value_counts())
    """
    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=random_state
    )

    features, labels = sampler.fit_resample(features, labels)

    return features, labels


def resample(
    features: pd.DataFrame,
    labels: pd.Series,
    method: Literal["under", "over"] | None = "under",
    sampling_strategy: str | float = "auto",
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample features and labels according to the chosen balancing method.

    Dispatches to the appropriate resampling strategy based on ``method``.
    Pass ``None`` when the dataset is already balanced to avoid discarding
    majority-class samples or introducing duplicates.

    Args:
        features (pd.DataFrame): Feature matrix with training samples.
        labels (pd.Series): Binary labels Series (0=non-eruption, 1=eruption).
        method (Literal["under", "over"] | None, optional): Resampling strategy.
            ``"under"`` applies ``RandomUnderSampler``, ``"over"`` applies
            ``RandomOverSampler``, and ``None`` returns the data unchanged.
            Defaults to ``"under"``.
        sampling_strategy (str | float, optional): Sampling ratio forwarded to
            the chosen sampler. Ignored when ``method=None``. Defaults to ``"auto"``.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
        verbose (bool, optional): Verbose mode. Defaults to ``False``.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple of (features, labels) after resampling.

    Examples:
        >>> X, y = resample(features, labels, method=None)
        >>> X, y = resample(features, labels, method="under", sampling_strategy=0.75)
        >>> X, y = resample(features, labels, method="over", random_state=0)
    """
    if method is None:
        if verbose:
            logger.info(
                "Resampling skipped (method=None) — dataset treated as balanced."
            )
        return features, labels

    if method == "under":
        if verbose:
            logger.info(
                f"Applying RandomUnderSampler (sampling_strategy={sampling_strategy})."
            )

        return random_under_sampler(
            features=features,
            labels=labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    if verbose:
        logger.info(
            f"Applying RandomOverSampler (sampling_strategy={sampling_strategy})."
        )

    sampler = RandomOverSampler(
        sampling_strategy=sampling_strategy, random_state=random_state
    )
    features, labels = sampler.fit_resample(features, labels)
    return features, labels


def load_features_resampled(
    features: pd.DataFrame | str,
    resampled: pd.DataFrame | pd.Series | str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct a per-seed resampled ``(X, y)`` pair from a labels-only payload.

    Slices the full feature matrix down to the ids recorded in ``resampled``,
    so the per-seed resampling decision can be replayed without persisting a
    copy of the feature matrix per seed. Duplicated ids (produced by
    over-sampling) are preserved in the returned ``X`` and ``y``.

    Args:
        features (pd.DataFrame | str): Full id-indexed feature matrix produced
            by :meth:`TrainingModel.extract_features` or
            :meth:`TrainingModel.load_features`. Pass a path to a
            ``features-matrix_*.parquet`` to load it on demand.
        resampled (pd.DataFrame | pd.Series | str): Per-seed resampled label
            payload. Accepts a path to ``features/{cv}/resampled/{seed}.csv``,
            a DataFrame carrying an ``is_erupted`` column, or a Series already
            indexed by id.
        columns (list[str] | None): Optional column projection — typically
            the seed's ``top_n_features``. ``None`` returns every feature
            column. Defaults to ``None``.

    Returns:
        tuple[pd.DataFrame, pd.Series]: ``(features_resampled,
            labels_resampled)`` aligned on the resampled id index.

    Examples:
        >>> X, y = load_features_resampled(
        ...     features=model.features_df,
        ...     resampled="training/features/.../resampled/00042.csv",
        ...     columns=top_n_features,
        ... )
    """
    if isinstance(features, str):
        features = pd.read_parquet(features)

    if isinstance(resampled, str):
        labels = pd.read_csv(resampled, index_col=0)["is_erupted"]
    elif isinstance(resampled, pd.DataFrame):
        labels = resampled["is_erupted"]
    else:
        labels = resampled

    idx = labels.index
    if columns is None:
        return features.loc[idx], labels
    return features.loc[idx, columns], labels


def get_significant_features(
    features: pd.DataFrame,
    labels: pd.Series | pd.DataFrame,
    fdr_level: float = 0.05,
    top_n: int = 20,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Get significant features ranked by p-value using tsfresh FeatureSelector.

    Uses tsfresh's FeatureSelector with Benjamini-Hochberg FDR correction to identify
    features with statistically significant correlation to the target labels. This is
    the first stage of feature selection in the pipeline.

    Args:
        features (pd.DataFrame): Extracted features DataFrame from tsfresh.
        labels (pd.Series | pd.DataFrame): Binary eruption labels. If DataFrame,
            will extract "is_erupted" column.
        fdr_level (float, optional): False discovery rate threshold (0.0-1.0).
            Lower values are more conservative. Defaults to 0.05.
        top_n (int, optional): Number of top features to return. Defaults to 20.
        n_jobs (int, optional): Number of parallel jobs for computation. Defaults to 1.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - features_filtered (pd.DataFrame): Filtered features DataFrame with only
              significant features.
            - significant_features (pd.Series): Features sorted by p-value (ascending),
              with feature names as index and p-values as values. Index name is "features",
              series name is "p_values".

    Examples:
        >>> filtered_features, sig_features = get_significant_features(
        ...     features_df, labels_series, fdr_level=0.05, n_jobs=4
        ... )
        >>> top_10_features = sig_features.head(10).index.tolist()
        >>> print(f"Selected {len(filtered_features.columns)} significant features")
    """
    if isinstance(labels, pd.DataFrame):
        labels = to_series(labels, column_value="is_erupted")

    selector = FeatureSelector(
        n_jobs=n_jobs, fdr_level=fdr_level, ml_task="classification"
    )

    # Extracted features with potentially reduced column
    features_filtered: pd.DataFrame = selector.fit_transform(X=features, y=labels)

    _significant_features: pd.Series = pd.Series(
        selector.p_values, index=selector.features
    )
    _significant_features = _significant_features.sort_values()
    _significant_features.name = "p_values"
    _significant_features.index.name = "features"

    # If no relevant features found, fall back to the 20 most significant
    # features ranked by p-value rather than FDR threshold.
    if len(features_filtered.columns) < top_n:
        selected_features: list[str] = _significant_features.head(top_n).index.tolist()

        if not selected_features:
            raise ValueError(
                "tsfresh found no significant features and the p-value fallback "
                "also returned nothing. The feature matrix may be empty, constant, "
                "or entirely uncorrelated with the labels. "
                f"Feature matrix shape: {features.shape}, "
                f"label distribution: {dict(labels.value_counts())}."
            )

        logger.warning(
            f"Significant features {len(features_filtered.columns)} less than {top_n}. "
            f"Using top {len(selected_features)} features (p-value based)."
        )

        features_filtered = features[selected_features]

    return features_filtered, _significant_features


def get_classifier_models(
    classifiers: list[str],
    cv_strategy: Literal[
        "shuffle", "stratified", "shuffle-stratified", "timeseries"
    ] = "shuffle-stratified",
    cv_splits: int = 5,
    verbose: bool = False,
) -> list[ClassifierModel]:
    """Instantiate one ClassifierModel per classifier key.

    Builds a :class:`~eruption_forecast.model.classifier_model.ClassifierModel`
    for each slug in ``classifiers``, applying a shared CV strategy and split count.

    Args:
        classifiers (list[str]): List of classifier slug names (e.g. ``["rf", "xgb"]``).
        cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified", "timeseries"],
            optional): Cross-validation strategy. Defaults to ``"shuffle-stratified"``.
        cv_splits (int, optional): Number of CV folds. Defaults to 5.
        verbose (bool, optional): Emit progress log messages. Defaults to False.

    Returns:
        list[ClassifierModel]: One configured :class:`ClassifierModel` per slug,
            in the same order as ``classifiers``.
    """
    classifier_models: list[ClassifierModel] = [
        ClassifierModel(
            classifier=classifier,  # ty:ignore[invalid-argument-type]
            cv_strategy=cv_strategy,
            n_splits=cv_splits,
            verbose=verbose,
        )
        for classifier in classifiers
    ]

    return classifier_models


def grid_search_cv(
    random_state: int,
    features: pd.DataFrame,
    labels: pd.Series,
    top_n_features: list[str],
    classifier_model: ClassifierModel,
    n_grids: int = 1,
    scoring: str = "balanced_accuracy",
) -> tuple[ClassifierModel, GridSearchCV, Any]:
    """Run GridSearchCV for a single seed and return the fitted results.

    Seeds the classifier model with ``random_state``, constructs a
    ``GridSearchCV`` over the model's parameter grid and CV splitter, fits it
    on the top-N selected features, and returns the configured classifier, the
    fitted search object, and the best estimator.

    Args:
        random_state (int): Random seed used to initialise the classifier.
        features (pd.DataFrame): Full feature matrix; only ``top_n_features``
            columns are passed to the search.
        labels (pd.Series): Binary target labels aligned with ``features``.
        top_n_features (list[str]): Column names of the pre-selected features
            to use during training.
        classifier_model (ClassifierModel): Configured classifier wrapper that
            exposes the estimator, parameter grid, and CV splitter.
        n_grids (int, optional): Number of parallel jobs for ``GridSearchCV``.
            Defaults to 1.
        scoring (str, optional): Scoring GridSearchCV. Defaults to "balanced_accuracy".
            See here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

    Returns:
        tuple[ClassifierModel, GridSearchCV, Any]: A 3-tuple of:
            - classifier: The seeded ``ClassifierModel`` instance.
            - grid_search: The fitted ``GridSearchCV`` object.
            - best_estimator: ``grid_search.best_estimator_``, i.e. the
              estimator retrained on the full training set with optimal params.
    """
    classifier: ClassifierModel = classifier_model.set_random_state(
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=classifier.model,
        param_grid=classifier.grid,
        cv=classifier.get_cv_splitter(),
        scoring=scoring,
        n_jobs=n_grids,
        verbose=0,
    )

    with joblib.parallel_backend("loky"):
        grid_search.fit(features[top_n_features], labels)

    return classifier, grid_search, grid_search.best_estimator_


def save_model_json(
    seeds: int,
    records: list[dict],
    classifier_dir: str,
    classifier_model: ClassifierModel,
    number_of_features: int,
    prefix_filename: str = "trained-model",
    verbose: bool = False,
) -> str:
    """Build and save the trained-models registry JSON for one classifier.

    Writes a list of per-seed records to a JSON file inside ``classifier_dir``.
    Each record contains ``random_state``, ``features`` (inline list of top-N
    feature names), and ``model_filepath``. Inlining the feature list lets
    :meth:`SeedEnsemble.from_json` rebuild the ensemble with a single registry
    read instead of one extra CSV per seed at load time.

    The filename encodes the classifier name, CV strategy, total seed count,
    and feature count so that downstream utilities can reconstruct all metadata
    from the path alone.

    Args:
        seeds (int): Total number of random seeds used during training; written
            into the filename as ``seeds-{seeds}``.
        records (list[dict]): Per-seed result dicts. Each must contain
            ``random_state`` (int), ``features`` (list of column names), and
            ``model_filepath`` (str).
        classifier_dir (str): Directory in which to write the registry JSON.
        classifier_model (ClassifierModel): Trained classifier wrapper; its
            ``name`` and ``cv_name`` attributes form the filename prefix.
        number_of_features (int): Number of top significant features selected;
            written into the filename as ``features-{number_of_features}``.
        prefix_filename (str, optional): Filename prefix for the saved
            registry. Defaults to ``"trained-model"``.
        verbose (bool, optional): Emit a log line on save. Defaults to ``False``.

    Returns:
        str: Absolute path to the saved registry JSON file.

    Raises:
        ValueError: If ``records`` is empty (no models were successfully
            trained), which would produce an empty registry.
    """
    if not records:
        raise ValueError("No significant features or trained models found.")

    classifier_id = f"{classifier_model.name}_{classifier_model.cv_name}"
    suffix = f"{classifier_id}_seeds-{seeds}_features-{number_of_features}"
    filename = f"{prefix_filename}__{suffix}.json"

    ensure_dir(classifier_dir)
    json_path = os.path.join(classifier_dir, filename)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    if verbose:
        logger.info(f"{classifier_model.name}: JSON trained model saved to {json_path}")

    return json_path


def compute_seed(
    seed_ensemble: SeedEnsemble,
    X: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    output_dir: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute per-seed classification metrics against ``y_true``.

    Runs :meth:`SeedEnsemble.compute_probabilities_and_predictions` once on
    ``X``, then iterates over each seed to build a per-seed metrics row and
    writes the aggregated table to
    ``{output_dir}/{classifier_name}/metrics/{classifier_name}_metrics_seeds-{n_seeds}.csv``.

    Args:
        seed_ensemble (SeedEnsemble): Fitted ensemble whose seeds are evaluated.
        X (pd.DataFrame): Extracted features DataFrame of shape
            ``(n_samples, n_features)``.
        y_true (np.ndarray | pd.Series): Ground-truth binary labels aligned
            positionally with ``X``. Length must equal ``n_samples``.
        output_dir (str): Root directory under which the per-classifier
            ``metrics/`` sub-directory and CSV are written. Created if missing.
        verbose (bool, optional): Log the saved CSV path when ``True``.
            Defaults to ``False``.

    Returns:
        tuple[pd.DataFrame, np.ndarray, np.ndarray]: A 3-tuple of:
            - metrics_df: One row per seed, indexed by ``random_state``, with
              columns ``accuracy``, ``balanced_accuracy``, ``precision``,
              ``average_precision``, ``pr_auc``, ``recall``, ``specificity``,
              ``f1_score``, ``roc_auc``, ``mcc``, ``g_mean``,
              ``true_positives``, ``false_positives``, ``true_negatives``,
              ``false_negatives``. ``pr_auc`` is the trapezoidal area under the
              precision-recall curve computed via
              :func:`sklearn.metrics.precision_recall_curve` + :func:`auc`;
              ``average_precision`` is the weighted-mean variant from
              :func:`sklearn.metrics.average_precision_score`.
            - y_probas: Probability matrix of shape ``(n_samples, n_seeds)``
              produced by the ensemble.
            - y_preds: Hard-label matrix of shape ``(n_samples, n_seeds)``
              produced by the ensemble (probas thresholded at 0.5 or per-seed
              calibrated threshold).

    Raises:
        ValueError: If ``len(y_true) != len(X)``.
    """
    classifier_name = seed_ensemble.classifier_name
    seeds = seed_ensemble.seeds

    metrics_dir = os.path.join(
        output_dir,
        classifier_name,
        "metrics",
    )

    save_filepath = os.path.join(
        metrics_dir,
        f"{classifier_name}_metrics_seeds-{len(seed_ensemble)}.csv",
    )

    y_true = np.asarray(y_true)
    if y_true.shape[0] != X.shape[0]:
        raise ValueError(
            f"y_true length ({y_true.shape[0]}) does not match X length ({X.shape[0]})."
        )

    y_probas, y_preds = seed_ensemble.compute_probabilities_and_predictions(X)

    rows: list[dict[str, float | int]] = []
    for idx, seed in enumerate(seeds):
        y_proba = y_probas[:, idx]
        y_pred = y_preds[:, idx]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
        recall = round(recall_score(y_true, y_pred, zero_division=0), 2)
        precision = round(precision_score(y_true, y_pred, zero_division=0), 2)

        precisions_curve, recalls_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(auc(recalls_curve, precisions_curve))

        metrics: dict[str, float | int] = {
            "random_state": int(seed["random_state"]),
            "accuracy": round(accuracy_score(y_true, y_pred), 2),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 2),
            "precision": precision,
            "average_precision": round(average_precision_score(y_true, y_proba), 2),
            "pr_auc": round(pr_auc, 2),
            "recall": recall,
            "specificity": round(specificity, 2),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 2),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 2),
            "mcc": round(matthews_corrcoef(y_true, y_pred), 2),
            "g_mean": round(float(np.sqrt(recall * specificity)), 2),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("random_state")

    ensure_dir(metrics_dir)
    df.to_csv(save_filepath, index=True)
    if verbose:
        logger.info(f"Saved {classifier_name} metrics: {save_filepath}")

    return df, y_probas, y_preds


def build_classifier_ensemble_summary(
    seed_ensemble: SeedEnsemble,
    labels: pd.Series | pd.DataFrame,
    eruption_dates: list[str],
) -> ClassifierEnsembleSummary:
    """Walk every seed × eruption-window and assemble the per-classifier summary.

    For each eruption date the day window ``[00:00, 23:59:59]`` is sliced from
    the per-seed probability matrix cached on
    :attr:`SeedEnsemble.probabilities`. Within that slice each seed contributes
    one :class:`SeedSummary` (highest + lowest probability rows). The
    across-windows extrema are tracked in a single pass and exposed on
    :attr:`ClassifierEnsembleSummary.highest` / ``.lowest`` so callers do not
    have to re-scan the structure to find the row to render.

    Args:
        seed_ensemble (SeedEnsemble): Fitted seed ensemble whose
            :attr:`probabilities` attribute has been populated by a prior
            ``predict_with_uncertainty`` / ``save_matrices`` call.
        labels (pd.Series | pd.DataFrame): Datetime-indexed label container
            from the upstream :class:`TrainingModel` / :class:`PredictionModel`
            — used to attach a ``datetime`` column to the probability matrix
            via :func:`to_datetime_index`.
        eruption_dates (list[str]): Eruption dates in ``YYYY-MM-DD`` form.
            Dates that do not intersect the prediction grid are skipped.

    Returns:
        ClassifierEnsembleSummary: Per-classifier summary.

    Raises:
        RuntimeError: If ``seed_ensemble.probabilities`` is ``None`` — the
            ensemble has not run a prediction yet.
    """
    if seed_ensemble.probabilities is None:
        raise RuntimeError(
            f"{seed_ensemble.classifier_name}: SeedEnsemble has no cached "
            f"probabilities. Run predict_with_uncertainty / forecast first."
        )

    df_probas = to_datetime_index(
        datetime_map=labels.sort_index(),
        df=seed_ensemble.probabilities,
    ).reset_index()

    summary = ClassifierEnsembleSummary(classifier_name=seed_ensemble.classifier_name)

    for eruption_date in eruption_dates:
        start_date = to_datetime(eruption_date).replace(hour=0, minute=0, second=0)
        end_date = start_date.replace(hour=23, minute=59, second=59)
        window = df_probas[
            (df_probas["datetime"] >= start_date) & (df_probas["datetime"] <= end_date)
        ]
        if window.empty:
            continue

        seed_summaries: list[SeedSummary] = []
        window_highest: ProbabilityPick | None = None
        window_lowest: ProbabilityPick | None = None
        for seed in seed_ensemble.seeds:
            random_state = int(seed["random_state"])
            column_name = f"seed_{random_state:05d}"

            sorted_window = window.sort_values(column_name, ascending=False)
            top_row = sorted_window.iloc[0]
            bottom_row = sorted_window.iloc[-1]

            seed_summary = SeedSummary(
                random_state=random_state,
                highest=ProbabilityPick(
                    random_state=random_state,
                    index=int(sorted_window.index[0]),
                    datetime=top_row["datetime"],
                    value=float(top_row[column_name]),
                ),
                lowest=ProbabilityPick(
                    random_state=random_state,
                    index=int(sorted_window.index[-1]),
                    datetime=bottom_row["datetime"],
                    value=float(bottom_row[column_name]),
                ),
            )
            seed_summaries.append(seed_summary)

            # Summarize per eruption date
            if (
                window_highest is None
                or seed_summary.highest.value > window_highest.value
            ):
                window_highest = seed_summary.highest
            if window_lowest is None or seed_summary.lowest.value < window_lowest.value:
                window_lowest = seed_summary.lowest

            # Summarize per classifier
            if (
                summary.highest is None
                or seed_summary.highest.value > summary.highest.value
            ):
                summary.highest = seed_summary.highest
            if (
                summary.lowest is None
                or seed_summary.lowest.value < summary.lowest.value
            ):
                summary.lowest = seed_summary.lowest

        if window_highest is None or window_lowest is None:
            raise RuntimeError(
                f"{seed_ensemble.classifier_name}: window for {eruption_date} "
                f"contributed no seed summaries — unreachable under the "
                f"upstream ``window.empty`` guard."
            )

        summary.eruption_windows.append(
            EruptionWindow(
                eruption_date=eruption_date,
                highest=window_highest,
                lowest=window_lowest,
                seeds=seed_summaries,
            )
        )

    return summary
