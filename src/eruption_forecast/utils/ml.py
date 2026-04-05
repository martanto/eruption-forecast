"""Machine learning utilities for model training, inference, and ensemble management.

This module centralises the ML-specific helpers that are shared across
``ModelTrainer``, ``ModelPredictor``, and the ensemble classes.  It covers
the full lifecycle from data preparation through prediction and model merging.

Key functions
-------------
- ``random_under_sampler`` — apply ``RandomUnderSampler`` to balance training data;
  returns resampled ``(X, y)`` and the fitted sampler
- ``get_significant_features`` — run tsfresh ``FeatureSelector`` (and optionally a
  RandomForest importance filter) to reduce the feature matrix to the most
  relevant columns
- ``compute_threshold_metrics`` — sweep decision thresholds and record precision,
  recall, F1, and balanced accuracy at each step
- ``compute_seed_eruption_probability`` — run one seed model on a feature matrix,
  aggregate per-seed statistics, and optionally persist a CSV
- ``compute_model_probabilities`` — iterate over all seeds in a ``SeedEnsemble`` or
  dict thereof and return a consensus probability DataFrame
- ``merge_seed_models`` — load all per-seed model files recorded in a registry CSV
  and bundle them into a ``SeedEnsemble``
- ``merge_all_classifiers`` — combine multiple ``SeedEnsemble`` objects into a single
  ``ClassifierEnsemble``
- ``get_classifier_models`` — instantiate ``ClassifierModel`` objects for a given
  classifier name and CV strategy
- ``load_labels_from_csv`` — thin wrapper around ``load_label_csv`` kept for
  backward compatibility
- ``get_classifier_label`` — translate an internal classifier key to a
  human-readable display label
"""

import os
from typing import Literal

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from tsfresh.transformers import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import (
    save_forecast_seed,
    predict_proba_from_estimator,
)
from eruption_forecast.model.constants import GPU_CLASSIFIERS
from eruption_forecast.utils.dataframe import to_series, load_label_csv
from eruption_forecast.config.constants import THRESHOLD_RESOLUTION
from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    resolution: int = THRESHOLD_RESOLUTION,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Sweep decision thresholds and compute classification metrics at each step.

    Iterates over ``resolution`` evenly-spaced thresholds from 0.0 to 1.0,
    binarises ``y_proba`` at each step, and records precision, recall, F1,
    and balanced accuracy. This is the single source of truth for threshold
    analysis used by both ``MetricsComputer`` and ``plot_threshold_analysis``.

    Args:
        y_true (np.ndarray): Ground-truth binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        resolution (int, optional): Number of threshold steps. Defaults to
            ``THRESHOLD_RESOLUTION``.

    Returns:
        tuple[np.ndarray, dict[str, list[float]]]: A 2-tuple of:
            - thresholds: 1-D array of length ``resolution`` from 0.0 to 1.0.
            - metrics_dict: dict with keys ``"precision"``, ``"recall"``,
              ``"f1"``, ``"balanced_accuracy"``, and ``"specificity"``,
              each a list of floats.
    """
    thresholds = np.linspace(0.0, 1.0, resolution)
    metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "balanced_accuracy": [],
        "specificity": [],
    }
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        metrics["f1"].append(f1_score(y_true, y_pred_thresh, zero_division=0))
        metrics["recall"].append(recall_score(y_true, y_pred_thresh, zero_division=0))
        metrics["precision"].append(
            precision_score(y_true, y_pred_thresh, zero_division=0)
        )
        metrics["balanced_accuracy"].append(
            balanced_accuracy_score(y_true, y_pred_thresh)
        )
        tn, fp, _, _ = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1]).ravel()
        metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return thresholds, metrics


def compute_g_mean(metrics: dict[str, list[float]]) -> np.ndarray:
    """Compute G-mean from threshold metrics.

    G-mean is the geometric mean of sensitivity (recall) and specificity.
    It is preferred over F1 for rare-event forecasting because it equally
    penalizes missing eruptions and false alarms without class-imbalance inflation.

    Args:
        metrics (dict[str, list[float]]): Metrics dict returned by
            ``compute_threshold_metrics``, must contain ``"recall"`` and
            ``"specificity"`` keys.

    Returns:
        np.ndarray: G-mean values, one per threshold step.
    """
    return np.sqrt(np.array(metrics["recall"]) * np.array(metrics["specificity"]))


def load_labels_from_csv(label_features_csv: str) -> pd.Series:
    """Load a label CSV and return a Series indexed by window ID.

    Delegates to :func:`eruption_forecast.utils.dataframe.load_label_csv`.
    Kept here for backward compatibility with existing call sites.

    Args:
        label_features_csv (str): Path to the label CSV file. Must contain
            an ``id`` column and an ``is_erupted`` column.

    Returns:
        pd.Series: Binary eruption labels indexed by window ID.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return load_label_csv(label_features_csv)


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

    _significant_features = pd.Series(selector.p_values, index=selector.features)
    _significant_features = _significant_features.sort_values()
    _significant_features.name = "p_values"
    _significant_features.index.name = "features"

    # If no relevant features found, fall back to the 20 most significant
    # features ranked by p-value rather than FDR threshold.
    if len(features_filtered.columns) < top_n:
        logger.warning(
            f"Significant features {len(features_filtered.columns)} less than {top_n}. Use top {top_n} features (p_values based)"
        )
        selected_features: list[str] = _significant_features.head(top_n).index.tolist()
        features_filtered = features[selected_features]

    return features_filtered, _significant_features


def compute_seed_eruption_probability(
    random_state: int,
    features_df: pd.DataFrame,
    significant_features_csv: str,
    model_filepath: str,
    output_dir: str | None = None,
    save: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eruption probability for a single random seed model.

    Loads a trained model and computes eruption probabilities for the given features.
    Supports both predict_proba (probabilistic) and decision_function (SVM) methods.
    Can cache results to disk for faster repeated predictions.

    Args:
        random_state (int): Random seed identifying the model.
        features_df (pd.DataFrame): Extracted features DataFrame for prediction.
        significant_features_csv (str): Path to CSV containing significant feature names.
        model_filepath (str): Path to the saved model (.pkl file).
        output_dir (str | None, optional): Directory to save predictions. If None,
            uses "output/predictions/seeds". Defaults to None.
        save (bool, optional): If True, save probabilities to CSV. Defaults to False.
        overwrite (bool, optional): If True, overwrite existing cached predictions.
            Defaults to False.
        verbose (bool, optional): If True, log save operations. Defaults to False.
        debug (bool, optional): If True, log detailed debug information. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - probabilities_eruption (np.ndarray): 1D array of eruption probabilities (P(class=1)).
            - probabilities_scores (np.ndarray): 2D array of shape (n_windows, 2) with
              columns [P(non-eruption), P(eruption)].
            - predictions_eruption (np.ndarray): 1D array of binary predictions (0 or 1).

    Raises:
        ValueError: If model output is 1-dimensional.
        RuntimeError: If model supports neither predict_proba nor decision_function.

    Examples:
        >>> proba_1d, proba_2d, pred_1d = compute_seed_eruption_probability(
        ...     random_state=42,
        ...     features_df=features,
        ...     significant_features_csv="sig_features.csv",
        ...     model_filepath="model_00042.pkl",
        ...     save=True
        ... )
        >>> print(proba_1d.mean())
        0.35
    """
    output_dir = output_dir or os.path.join(os.getcwd(), "output", "predictions")
    output_dir = os.path.join(output_dir, "seeds")

    filename = f"p_{random_state:05d}.csv"
    filepath = os.path.join(output_dir, f"{filename}")

    # Return cached result regardless of the `save` flag — if the file exists
    # and overwrite is False, we skip inference entirely and serve the cache.
    # `save=True` here means "persist if not already cached", not "always write".
    if os.path.exists(filepath) and not overwrite:
        seed_df = pd.read_csv(filepath, index_col=0)
        eruption_probabilities = seed_df["p_eruption"]
        eruption_predictions = seed_df["prediction"]
        return (
            eruption_probabilities.to_numpy(),
            seed_df[
                [
                    "p_non_eruption",
                    "p_eruption",
                ]
            ].to_numpy(),
            eruption_predictions.to_numpy(),
        )

    df_sig = pd.read_csv(significant_features_csv, index_col=0)
    feature_names: list[str] = df_sig.index.tolist()

    # Load trained model
    model = joblib.load(model_filepath)

    # Select features dataframe with top-n significant features
    X: pd.DataFrame = features_df[feature_names]

    probabilities_eruption, probabilities_scores, predictions_eruption = (
        predict_proba_from_estimator(model, X, identifier=random_state)
    )

    if debug:
        logger.debug(
            f"{random_state:05d} :: probabilities_eruption values: {probabilities_eruption}"
        )

    if save:
        save_forecast_seed(
            output_dir,
            random_state,
            probabilities_eruption,
            predictions_eruption,
            overwrite=overwrite,
            verbose=verbose,
        )

    return probabilities_eruption, probabilities_scores, predictions_eruption


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
    if basename.startswith("trained_model_"):
        return basename[len("trained_model_") :]
    return basename


def merge_seed_models(
    trained_model_csv: str,
    output_dir: str | None = None,
) -> str:
    """Load all seed models from a registry CSV and bundle into one SeedEnsemble pkl.

    Reads the trained-model registry CSV produced by ``ModelTrainer``, loads
    every seed estimator and its significant-feature list into memory, and
    serialises the resulting :class:`~eruption_forecast.model.seed_ensemble.SeedEnsemble`
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
    output_path = os.path.join(output_dir, f"merged_model_{suffix}.pkl")

    ensemble = SeedEnsemble.from_registry(trained_model_csv)
    ensemble.save(output_path)

    logger.info(f"Saved merged seed model to: {output_path}")

    return output_path


def merge_all_classifiers(
    trained_models: dict[str, str],
    output_path: str | None = None,
) -> str:
    """Merge multiple classifier registry CSVs into a single multi-classifier pkl.

    Calls :func:`merge_seed_models` for each classifier, bundles the resulting
    :class:`~eruption_forecast.model.seed_ensemble.SeedEnsemble` objects into a
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


def get_classifier_label(classifier_name: str) -> str:
    """Return a human-readable label for a classifier given its scikit-learn class name.

    Looks up ``classifier_name`` in a fixed mapping of class names to display labels.
    If the name is not found (e.g., an unrecognised or custom classifier), the input
    string is returned unchanged.

    Args:
        classifier_name (str): Scikit-learn class name of the classifier, e.g.
            ``"RandomForestClassifier"`` or ``"XGBClassifier"``.

    Returns:
        str: Human-readable display label, e.g. ``"Random Forest"`` or ``"XGBoost"``.
            Returns ``classifier_name`` unchanged if not found in the mapping.
    """
    classifier_slugs = {
        "SVC": "svm",
        "KNeighborsClassifier": "KNN",
        "DecisionTreeClassifier": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
        "LiteRandomForestClassifier": "(lite) Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "XGBClassifier": "XGBoost",
        "MLPClassifier": "Neural Network",
        "GaussianNB": "Naive Bayes",
        "LogisticRegression": "Logistic Regression",
        "VotingClassifier": "Voting Classifier",
    }

    if classifier_name not in classifier_slugs:
        return classifier_name

    return classifier_slugs[classifier_name]
