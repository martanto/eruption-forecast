import os
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
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from eruption_forecast.logger import logger
from eruption_forecast.model.constants import GPU_CLASSIFIERS
from eruption_forecast.utils.dataframe import to_series
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.utils.date_utils import sort_dates
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


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


def split_eruption_dates(
    eruption_dates: list[str] | list[datetime], test_size: float = 0.2
):
    """Split eruption dates into training and testing sets.

    Args:
        eruption_dates (list[str] | list[datetime]): Eruption dates.
        test_size (float, optional): Fraction of test data.
    """
    if test_size < 0 or test_size > 1:
        raise ValueError(f"test_size must be between 0 and 1. You provided {test_size}")

    eruption_dates = sort_dates(eruption_dates, as_datetime=False)

    # Clip to ensure not used all data as test
    split_idx = np.clip(
        np.ceil(len(eruption_dates) * test_size),
        a_min=1,
        a_max=len(eruption_dates) - 1,
    ).astype(int)

    X_train = eruption_dates[:-split_idx]
    X_test = eruption_dates[-split_idx:]
    return X_train, X_test


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    resolution: int = 101,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Sweep decision thresholds and compute classification metrics at each step.

    Iterates over ``resolution`` evenly-spaced thresholds from 0.0 to 1.0,
    binarises ``y_proba`` at each step, and records precision, recall, F1,
    balanced accuracy, and MCC. This is the single source of truth for threshold
    analysis used by both ``MetricsComputer`` and ``plot_threshold_analysis``.

    Args:
        y_true (np.ndarray): Ground-truth binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        resolution (int, optional): Number of threshold steps. Defaults to ``101``.

    Returns:
        tuple[np.ndarray, dict[str, list[float]]]: A 2-tuple of:
            - thresholds: 1-D array of length ``resolution`` from 0.0 to 1.0.
            - metrics_dict: dict with keys ``"precision"``, ``"recall"``,
              ``"f1"``, ``"balanced_accuracy"``, ``"specificity"``, and
              ``"mcc"``, each a list of floats.
    """
    thresholds = np.linspace(0.0, 1.0, resolution)
    metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
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
            whose lists were never populated (e.g. ``"roc_auc"``) are skipped.

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

    if method == "over":
        if verbose:
            logger.info(
                f"Applying RandomOverSampler (sampling_strategy={sampling_strategy})."
            )

        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        features, labels = sampler.fit_resample(features, labels)
        return features, labels

    raise ValueError(
        f"Unknown resample method '{method}'. Choose from 'under', 'over', or None."
    )


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


def _extract_trained_model_suffix(csv_path: str) -> str:
    """Extract the suffix portion from a trained-model registry CSV filename.

    Strips the ``trained_model_`` prefix (if present) from the basename of
    ``csv_path`` and returns the remainder as a plain string.  Used by
    :func:`merge_seed_models` and :func:`merge_all_classifiers` to derive
    output filenames without duplicating the stripping logic.

    Args:
        csv_path (str): Path to a trained-model registry CSV file.

    Returns:
        str: The suffix after ``"trained_model_"``, or the full basename
            (without extension) if the prefix is absent.
    """
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    if basename.startswith("trained-model_"):
        return basename[len("trained-model_") :]
    return basename


def merge_seed_models(
    trained_model_csv: str,
    output_dir: str | None = None,
) -> str:
    """Load all seed models from a registry CSV and bundle into one SeedEnsemble pkl.

    Reads the trained-model registry CSV produced by ``ModelTrainer``, loads
    every seed estimator and its significant-feature list into memory, and
    serialises the resulting :class:`~eruption_forecast.ensemble.seed_ensemble.SeedEnsemble`
    to a single ``.pkl`` file.  This eliminates the per-seed I/O overhead at
    prediction time.

    Args:
        trained_model_csv (str): Path to the trained-model registry CSV (the
            ``trained_model_{suffix}.csv`` file written by ``ModelTrainer``).
        output_dir (str | None, optional): Destination path for the merged
            ``.pkl`` file.  If ``None``, the file is written to the same
            directory as ``trained_model_csv`` with the name
            ``merged_model_{suffix}.pkl``, where ``{suffix}`` is derived from
            the registry CSV filename.  Defaults to ``None``.

    Returns:
        str: Absolute path to the saved merged ``.pkl`` file.

    Raises:
        FileNotFoundError: If ``registry_csv`` does not exist.
    """
    output_dir = (
        output_dir
        if output_dir is not None
        else os.path.dirname(os.path.abspath(trained_model_csv))
    )

    suffix = _extract_trained_model_suffix(trained_model_csv)
    output_path = os.path.join(output_dir, f"SeedEnsemble_{suffix}.pkl")

    ensemble = SeedEnsemble.from_registry(trained_model_csv)
    ensemble.save(output_path)

    logger.info(f"Saved SeedEnsemble model to: {output_path}")

    return output_path


def merge_all_classifiers(
    trained_models: dict[str, str],
    output_path: str | None = None,
) -> str:
    """Merge multiple classifier registry CSVs into a single multi-classifier pkl.

    Calls :func:`merge_seed_models` for each classifier, bundles the resulting
    :class:`~eruption_forecast.ensemble.seed_ensemble.SeedEnsemble` objects into a
    plain ``dict[str, SeedEnsemble]``, and serialises the dict to one ``.pkl``
    file.  ``ModelPredictor`` detects this dict type automatically when the path
    is passed as the ``trained_models`` parameter.

    Args:
        trained_models (dict[str, str]): Mapping of classifier name to the path
            of its trained-model registry CSV (e.g.
            ``{"rf": "path/to/rf_registry.csv", "xgb": "path/to/xgb_registry.csv"}``).
        output_path (str | None, optional): Destination path for the combined
            ``.pkl`` file.  If ``None``, the file is placed one directory above
            the first registry CSV (i.e. in the ``trainings/`` root) and named
            ``merged_classifiers_{suffix}.pkl``, where ``{suffix}`` is derived
            from the first registry CSV filename.  Defaults to ``None``.

    Returns:
        str: Absolute path to the saved combined ``.pkl`` file.

    Raises:
        ValueError: If ``trained_models`` is empty.
        FileNotFoundError: If any registry CSV does not exist.
    """
    if not trained_models:
        raise ValueError("trained_models must not be empty.")

    if output_path is None:
        first_csv = next(iter(trained_models.values()))
        # Go one level up from the classifier dir to the trainings root
        trainings_dir = os.path.dirname(os.path.dirname(os.path.abspath(first_csv)))
        suffix = _extract_trained_model_suffix(first_csv)
        output_path = os.path.join(trainings_dir, f"merged_classifiers_{suffix}.pkl")

    ensembles: dict[str, SeedEnsemble] = {}
    for name, csv_path in trained_models.items():
        logger.info(f"[merge_all_classifiers] Merging classifier: {name}")
        ensembles[name] = SeedEnsemble.from_registry(csv_path)

    classifier_ensemble = ClassifierEnsemble.from_seed_ensembles(ensembles)
    classifier_ensemble.save(output_path)

    logger.info(f"Saved merged classifier model to: {output_path}")

    return output_path


def get_default_features() -> list[str]:
    """Return the sorted list of tsfresh ComprehensiveFCParameters feature names.

    Instantiates ``ComprehensiveFCParameters`` and extracts its keys, giving the
    full set of features that tsfresh can compute. See the tsfresh documentation
    for the complete feature catalogue.

    Returns:
        list[str]: Sorted list of tsfresh feature name strings.
    """
    default_fc_parameters = ComprehensiveFCParameters()
    default_fc_parameters_keys = default_fc_parameters.data
    keys: list[str] = list(default_fc_parameters_keys.keys())
    keys.sort()
    return keys


def get_classifier_models(
    classifiers: list[str],
    cv_strategy: Literal[
        "shuffle", "stratified", "shuffle-stratified", "timeseries"
    ] = "shuffle-stratified",
    cv_splits: int = 5,
    use_gpu: bool = False,
    gpu_id: int = 0,
    verbose: bool = False,
) -> list[ClassifierModel]:
    """Instantiate one ClassifierModel per classifier key.

    Builds a :class:`~eruption_forecast.model.classifier_model.ClassifierModel`
    for each slug in ``classifiers``, applying a shared CV strategy, split count,
    and GPU settings. GPU acceleration is enabled only for classifiers listed in
    ``GPU_CLASSIFIERS``.

    Args:
        classifiers (list[str]): List of classifier slug names (e.g. ``["rf", "xgb"]``).
        cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified", "timeseries"],
            optional): Cross-validation strategy. Defaults to ``"shuffle-stratified"``.
        cv_splits (int, optional): Number of CV folds. Defaults to 5.
        use_gpu (bool, optional): Enable GPU acceleration for supported classifiers.
            Defaults to False.
        gpu_id (int, optional): GPU device index when ``use_gpu`` is True. Defaults to 0.
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
            use_gpu=use_gpu and classifier in GPU_CLASSIFIERS,
            gpu_id=gpu_id,
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


def save_model_csv(
    seeds: int,
    records: list[dict],
    classifier_dir: str,
    classifier_model: ClassifierModel,
    number_of_features: int,
    prefix_filename: str = "trained-model",
    verbose: bool = False,
) -> str:
    """Build and save the trained-models registry CSV for one classifier.

    Constructs a ``pandas.DataFrame`` from ``records``, sets ``random_state``
    as the index, and writes it to a CSV file inside ``classifier_dir``. The
    filename encodes the classifier name, CV strategy, total seed count, and
    feature count so that downstream utilities can reconstruct all metadata
    from the path alone.

    Args:
        seeds (int): Total number of random seeds used during training; written
            into the filename as ``rs-{seeds}``.
        records (list[dict]): List of per-seed result dicts, each containing at
            least a ``"random_state"`` key plus model and feature-path fields.
        classifier_dir (str): Directory in which to write the registry CSV.
        classifier_model (ClassifierModel): Trained classifier wrapper; its
            ``name`` and ``cv_name`` attributes form the filename prefix.
        number_of_features (int): Number of top significant features selected;
            written into the filename as ``top-{number_of_features}``.
        prefix_filename (str, optional): Filename prefix for saved training model CSV.
            Defaults to ``"trained_model"``.
        verbose (bool, optional): Show detailed information. Defaults to False.

    Returns:
        str: Absolute path to the saved registry CSV file.

    Raises:
        ValueError: If ``records`` is empty (no models were successfully
            trained), which would produce an empty registry.
    """
    classifier_id = f"{classifier_model.name}_{classifier_model.cv_name}"

    suffix = f"{classifier_id}_seeds-{seeds}_features-{number_of_features}"
    filename = f"{prefix_filename}__{suffix}.csv"

    registry_df = pd.DataFrame(records).set_index("random_state")
    if registry_df.empty:
        raise ValueError("No significant features or trained models found.")

    csv = os.path.join(classifier_dir, filename)
    registry_df.to_csv(csv, index=True)

    if verbose:
        logger.info(f"{classifier_model.name}: CSV trained model saved to {csv}")

    return csv


def compute_seed(
    seed_ensemble: SeedEnsemble,
    X: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    output_dir: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
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

    return df, y_probas
