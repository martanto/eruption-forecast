"""Machine learning utilities for model training and evaluation.

This module provides functions for data balancing, feature selection, metrics
computation, and eruption probability prediction.
"""

import os
import json
from typing import Any, cast

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from tsfresh.transformers import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV

from eruption_forecast.logger import logger
from eruption_forecast.utils.dataframe import to_series
from eruption_forecast.config.constants import ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.model.classifier_model import ClassifierModel


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

    return features_filtered, _significant_features


def get_metrics(
    classifier_model: ClassifierModel,
    labels_test,
    labels_pred,
    labels_train: pd.Series,
    top_n: int,
    grid_search: GridSearchCV,
    random_state: int,
    metrics_filepath: str | None = None,
) -> dict[str | Any, int | str | Any]:
    """Compute classification metrics from model predictions.

    Calculates confusion matrix components, accuracy, balanced accuracy,
    F1 score, precision, recall, and best cross-validation parameters.
    Optionally saves the metrics to a JSON file. This is the core metrics
    function used by ModelTrainer.

    Args:
        classifier_model (ClassifierModel): Fitted classifier model with name,
            cv_strategy, and n_splits attributes.
        labels_test (array-like): True test labels.
        labels_pred (array-like): Predicted labels from the model.
        labels_train (pd.Series): Training labels used for training.
        top_n (int): Number of features used in training.
        grid_search (GridSearchCV): Fitted GridSearchCV object with
            best_params_ and best_score_ attributes.
        random_state (int): Random state seed used for reproducibility.
        metrics_filepath (str | None, optional): Path to save metrics as
            a JSON file. If None, does not save. Defaults to None.

    Returns:
        dict[str | Any, int | str | Any]: Metrics dictionary with keys:
            - ``random_state`` (int): Random seed used.
            - ``classifier`` (str): Classifier name.
            - ``cv_strategy`` (str): Cross-validation strategy.
            - ``cv_splits`` (int): Number of CV splits.
            - ``n_train`` (int): Number of training samples.
            - ``n_test`` (int): Number of test samples.
            - ``n_features`` (int): Number of features used.
            - ``true_negatives`` (int): TN count.
            - ``false_positives`` (int): FP count.
            - ``false_negatives`` (int): FN count.
            - ``true_positives`` (int): TP count.
            - ``accuracy`` (float): Accuracy score.
            - ``balanced_accuracy`` (float): Balanced accuracy score.
            - ``f1_score`` (float): F1 score.
            - ``precision`` (float): Precision score.
            - ``recall`` (float): Recall score.
            - ``best_params`` (dict): Best hyperparameters from GridSearchCV.
            - ``best_cv_score`` (float): Best CV score from GridSearchCV.

    Examples:
        >>> metrics = get_metrics(
        ...     classifier_model=clf_model,
        ...     labels_test=y_test,
        ...     labels_pred=y_pred,
        ...     labels_train=y_train,
        ...     top_n=20,
        ...     grid_search=gs,
        ...     random_state=42,
        ...     metrics_filepath="output/metrics.json",
        ... )
        >>> print(f"F1: {metrics['f1_score']:.3f}")
        F1: 0.850
    """

    # Confusion matrix for Binary Classification
    # Read more: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#binary-classification
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels_test, labels_pred).ravel().tolist()
    )

    metrics = {
        "random_state": random_state,
        "classifier": classifier_model.name,
        "cv_strategy": classifier_model.cv_strategy,
        "cv_splits": classifier_model.n_splits,
        "n_train": len(labels_train),
        "n_test": len(labels_test),
        "n_features": top_n,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_positives": true_positive,
        "accuracy": accuracy_score(labels_test, labels_pred),
        "balanced_accuracy": balanced_accuracy_score(labels_test, labels_pred),
        "f1_score": f1_score(labels_test, labels_pred),
        "precision": precision_score(labels_test, labels_pred),
        "recall": recall_score(labels_test, labels_pred),
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
    }

    # Save metrics
    if metrics_filepath is not None:
        with open(metrics_filepath, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


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
) -> tuple[np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - probabilities_eruption (np.ndarray): 1D array of eruption probabilities (P(class=1)).
            - probabilities_scores (np.ndarray): 2D array of shape (n_windows, 2) with
              columns [P(non-eruption), P(eruption)].

    Raises:
        ValueError: If model output is 1-dimensional.
        RuntimeError: If model supports neither predict_proba nor decision_function.

    Examples:
        >>> proba_1d, proba_2d = compute_seed_eruption_probability(
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

    if os.path.exists(filepath) and not overwrite:
        seed_df = pd.read_csv(filepath, index_col=0)
        eruption_probabilities = seed_df["p_eruption"]
        return eruption_probabilities.to_numpy(), seed_df.to_numpy()

    df_sig = pd.read_csv(significant_features_csv, index_col=0)
    feature_names: list[str] = df_sig.index.tolist()

    # Load trained model
    model = joblib.load(model_filepath)

    # Select features dataframe with top-n significant features
    X = features_df[feature_names]

    if hasattr(model, "predict_proba"):
        # probabilities_scores has shape (n_rows, n_windows). Where n_window will reflect
        # the number of the classifiications. In this case n_windows is 2 which indicates
        # 0 as nnn-eruption, and 1 as an eruption.
        probabilities_scores: np.ndarray = model.predict_proba(X)

        if probabilities_scores.ndim == 1:
            raise ValueError(
                f"Your probability for seed {random_state} scores only has 1 dimensions. "
                f"It should have 2 dimensions which consists of `0` (non-eruption) and `1` (eruption)."
            )

        # Select column 1 as eruption probabilities representative. Column 0 is non-eruption.
        probabilities_eruption: np.ndarray = probabilities_scores[:, 1]

        if debug:
            logger.debug(f"{random_state:05d} :: predict_probe was used")
            logger.debug(
                f"{random_state:05d} :: probabilities_eruption values: {probabilities_eruption}"
            )

    elif hasattr(model, "decision_function"):
        probabilities_scores: np.ndarray = model.decision_function(X)
        probabilities_eruption: np.ndarray = 1.0 / (1.0 + np.exp(-probabilities_scores))

        if debug:
            logger.debug(f"{random_state:05d} :: decision_function was used")
            logger.debug(
                f"{random_state:05d} :: probabilities_eruption values: {probabilities_eruption}"
            )

    else:
        raise RuntimeError(
            f"Model at {model_filepath} supports neither "
            "predict_proba nor decision_function."
        )

    if save and not overwrite:
        os.makedirs(output_dir, exist_ok=True)
        probabilities_df = pd.DataFrame(
            probabilities_scores, columns=["p_non_eruption", "p_eruption"]
        )

        probabilities_df.index.name = "label_id"
        probabilities_df.to_csv(filepath, index=True)

        if verbose:
            logger.info(f"Saved seed {random_state:05d} probability to: {filepath}")

    return probabilities_eruption, probabilities_scores


def compute_model_probabilities(
    df_models: pd.DataFrame,
    features_df: pd.DataFrame,
    features_csv_column: str = "significant_features_csv",
    trained_model_filepath_column: str = "trained_model_filepath",
    classifier_name: str = "model",
    threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
    number_of_seeds: int | None = None,
    output_dir: str | None = None,
    save_predictions: bool = False,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate eruption probabilities across all seeds of a single classifier.

    Computes consensus predictions by averaging probabilities from multiple models
    trained with different random seeds. This reduces variance and improves prediction
    reliability. Calculates mean probability, uncertainty (std), confidence, and
    binary predictions.

    Args:
        df_models (pd.DataFrame): Model registry DataFrame with random_state as index.
            Must contain columns specified by features_csv_column and
            trained_model_filepath_column.
        features_df (pd.DataFrame): Extracted feature matrix for the prediction windows.
        features_csv_column (str, optional): Column name containing paths to significant
            features CSVs. Defaults to "significant_features_csv".
        trained_model_filepath_column (str, optional): Column name containing paths to
            trained model files. Defaults to "trained_model_filepath".
        classifier_name (str, optional): Classifier name for logging. Defaults to "model".
        threshold (float, optional): Minimum mean probability threshold to classify as
            eruption (0.0-1.0). Defaults to 0.7.
        number_of_seeds (int | None, optional): Maximum number of seeds to use. If None,
            uses all seeds. Defaults to None.
        output_dir (str | None, optional): Directory to save per-seed predictions.
            Defaults to None.
        save_predictions (bool, optional): If True, save per-seed predictions to CSV.
            Defaults to False.
        overwrite (bool, optional): If True, overwrite existing cached predictions.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays
            of shape (n_windows,):
            - mean_probability (np.ndarray): Mean eruption probability across seeds.
            - std_proba (np.ndarray): Standard deviation of probabilities (uncertainty).
            - confidence (np.ndarray): Voting agreement fraction (0.5-1.0).
            - prediction (np.ndarray): Binary predictions (0 or 1) based on threshold.

    Examples:
        >>> mean_p, std_p, conf, pred = compute_model_probabilities(
        ...     df_models=model_registry,
        ...     features_df=features,
        ...     threshold=0.6,
        ...     number_of_seeds=100
        ... )
        >>> print(f"Mean eruption probability: {mean_p.mean():.3f}")
        Mean eruption probability: 0.450
    """
    seed_eruption_probabilities: list[np.ndarray] = []

    for seed_count, (random_state, row) in enumerate(df_models.iterrows(), start=1):
        random_state_int = int(cast(int, random_state))
        significant_features_csv: str = row[features_csv_column]
        model_filepath: str = row[trained_model_filepath_column]

        probabilities_eruption, _ = compute_seed_eruption_probability(
            random_state=random_state_int,
            significant_features_csv=significant_features_csv,
            model_filepath=model_filepath,
            features_df=features_df,
            output_dir=output_dir,
            overwrite=overwrite,
            save=save_predictions,
        )

        seed_eruption_probabilities.append(probabilities_eruption)
        logger.debug(
            f"[{classifier_name}] Seed {random_state:05d} — "
            f"mean P(eruption): {probabilities_eruption.mean():.4f}"
        )

        if number_of_seeds is not None and seed_count >= number_of_seeds:
            break

    probabilities_eruption_matrix = np.stack(
        seed_eruption_probabilities, axis=1
    )  # (n_windows, n_seeds)

    mean_probability: np.ndarray = probabilities_eruption_matrix.mean(axis=1)
    std_proba: np.ndarray = probabilities_eruption_matrix.std(axis=1)
    prediction: np.ndarray = (mean_probability >= threshold).astype(int)

    votes_for_eruption: np.ndarray = (probabilities_eruption_matrix >= 0.5).sum(axis=1)
    n_seeds = probabilities_eruption_matrix.shape[0]
    majority_votes = np.where(
        prediction == 1, votes_for_eruption, n_seeds - votes_for_eruption
    )
    confidence: np.ndarray = majority_votes / n_seeds

    return mean_probability, std_proba, confidence, prediction
