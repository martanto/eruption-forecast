"""Array operations, outlier detection, and seed-probability aggregation utilities.

This module provides low-level array helpers used throughout the pipeline for
seismic data quality control, probabilistic inference, and per-seed output
management.

Key functions
-------------
- ``detect_maximum_outlier`` — z-score test on the array maximum; returns index and value
- ``remove_maximum_outlier`` — removes a single maximum outlier per call
- ``remove_outliers`` — removes all z-score outliers in one pass
- ``detect_anomalies_zscore`` — returns a boolean mask of anomalous positions using
  the modified z-score (MAD-based)
- ``mask_zero_values`` / ``filter_nans`` — array cleaning helpers
- ``count_valid_values`` / ``get_completeness`` — data-quality metrics for ObsPy
  streams and traces
- ``predict_proba_from_estimator`` — unified ``predict_proba`` / ``decision_function``
  interface; the single entry point for per-seed inference
- ``aggregate_seed_probabilities`` — reduces a seed × sample matrix to mean, std,
  confidence, and vote statistics
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from obspy import Trace, Stream
from numpy.typing import NDArray

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir


def _save_seed_proba_csv(
    output_dir: str,
    random_state: int,
    p_eruption: np.ndarray,
    predictions: np.ndarray,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Save per-seed eruption probabilities to a CSV file.

    Writes a CSV with columns ``p_non_eruption``, ``p_eruption``, and
    ``prediction`` to ``{output_dir}/p_{random_state:05d}.csv``.  Skips
    writing if the file already exists and ``overwrite`` is ``False``.

    Args:
        output_dir (str): Directory where the CSV will be written.
        random_state (int): Seed identifier used to build the filename.
        p_eruption (np.ndarray): 1-D array of P(eruption) values.
        predictions (np.ndarray): 1-D array of binary predictions (0 or 1).
        overwrite (bool, optional): If ``True``, overwrite an existing file.
            Defaults to ``False``.
        verbose (bool, optional): If ``True``, log a message after saving.
            Defaults to ``False``.
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"p_{random_state:05d}.csv")
    if os.path.exists(filepath) and not overwrite:
        return
    df = pd.DataFrame(
        np.column_stack((1 - p_eruption, p_eruption, predictions)),
        columns=["p_non_eruption", "p_eruption", "prediction"],
    )
    df.index.name = "label_id"
    df.to_csv(filepath, index=True)
    if verbose:
        logger.info(f"Saved seed {random_state:05d} probability to: {filepath}")


def predict_proba_from_estimator(
    model: Any,
    X: pd.DataFrame,
    identifier: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call ``predict_proba`` or ``decision_function`` on a fitted estimator.

    Abstracts away the two common sklearn output conventions — probabilistic
    classifiers (``predict_proba``) and margin-based classifiers
    (``decision_function``) — into a single interface.  The ``ndim`` of the
    ``predict_proba`` output is validated to guard against estimators that
    return a 1-D array instead of the expected ``(n_samples, 2)`` shape.

    For the ``decision_function`` case the raw scores are converted to
    probabilities via the logistic sigmoid, and a ``(n_samples, 2)`` scores
    array is reconstructed so both return values always have the same shape.

    Args:
        model (Any): Fitted sklearn-compatible estimator.
        X (pd.DataFrame): Feature subset to predict on.
        identifier (str | int | None, optional): Label used in error messages
            to identify the model (e.g. the random state or file path).
            Defaults to ``None``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A 3-tuple:

            - ``eruption_proba`` (np.ndarray): 1-D array of shape
              ``(n_samples,)`` with P(eruption).
            - ``scores`` (np.ndarray): 2-D array of shape ``(n_samples, 2)``
              with columns ``[P(non-eruption), P(eruption)]``.
            - ``eruption_predict`` (np.ndarray): 1-D array of shape
              ``(n_samples,)`` binary prediction (0 or 1).

    Raises:
        ValueError: If ``predict_proba`` returns a 1-D array.
        RuntimeError: If the estimator supports neither ``predict_proba``
            nor ``decision_function``.
    """
    identifier = str(identifier) if identifier is not None else "model"

    if hasattr(model, "predict_proba"):
        scores: np.ndarray = model.predict_proba(X)
        if scores.ndim == 1:
            raise ValueError(
                f"Probability scores for {identifier} have 1 dimension; "
                "expected 2 dimensions [P(non-eruption), P(eruption)]."
            )
        eruption_proba: np.ndarray = scores[:, 1]
        eruption_predict = model.predict(X)
    elif hasattr(model, "decision_function"):
        raw: np.ndarray = model.decision_function(X)
        eruption_proba = 1.0 / (1.0 + np.exp(-raw))
        scores = np.column_stack([1.0 - eruption_proba, eruption_proba])
        eruption_predict = (eruption_proba >= 0.5).astype(int)
    else:
        raise RuntimeError(
            f"{identifier} supports neither predict_proba nor decision_function."
        )

    return eruption_proba, scores, eruption_predict


def aggregate_seed_probabilities(
    seed_proba_matrix: np.ndarray,
    seed_predict_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a seed probability matrix into summary statistics.

    Computes the mean probability, standard deviation (uncertainty), confidence,
    and mean prediction from matrices of per-seed eruption probabilities and
    binary predictions. Confidence is a CI-like metric: ``1.96 * sqrt(p * (1-p) / n)``.

    This is the shared aggregation kernel used by both
    :meth:`SeedEnsemble.predict_with_uncertainty` and
    :func:`eruption_forecast.utils.ml.compute_model_probabilities`.

    Args:
        seed_proba_matrix (np.ndarray): Array of shape (n_samples, n_seeds)
            containing per-seed P(eruption) values.
        seed_predict_matrix (np.ndarray): Array of shape (n_samples, n_seeds)
            containing per-seed binary predictions (0 or 1).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four 1-D arrays
            of shape (n_samples,):

            - ``seed_probability``: Mean P(eruption) across seeds.
            - ``seed_uncertainty``: Standard deviation of P(eruption) across seeds.
            - ``seed_prediction``: Mean of per-seed binary votes (continuous, [0, 1]).
            - ``seed_confidence``: CI-like metric ``1.96 * sqrt(p * (1-p) / n_seeds)``.
    """
    seed_probability: np.ndarray = seed_proba_matrix.mean(axis=1)
    seed_uncertainty: np.ndarray = seed_proba_matrix.std(axis=1)
    seed_prediction: np.ndarray = seed_predict_matrix.mean(axis=1)

    n_seeds = seed_proba_matrix.shape[1]
    seed_confidence = 1.96 * (
        np.sqrt(seed_prediction * (1 - seed_prediction) / n_seeds)
    )

    return seed_probability, seed_uncertainty, seed_prediction, seed_confidence


def detect_anomalies_zscore(
    data: np.ndarray, threshold: float = 3.5
) -> NDArray[np.bool_]:
    """Detect anomalies using z-score method.

    Args:
        data (np.ndarray): Array of numerical data.
        threshold (float, optional): Z-score threshold. Defaults to 3.5.

    Returns:
        NDArray[np.bool_]: Boolean mask array of same length as ``data``.
            True indicates an anomaly at that position.
    """
    # Compute on non-NaN values only
    valid = filter_nans(data)
    median = np.median(valid)

    # Median Absolute Deviation
    mad = np.median(np.abs(valid - median))
    if mad == 0:
        return np.zeros(len(data), dtype=bool)

    # 0.6745 quantile of the standard normal distribution
    modified_z_score = 0.6745 * (data - median) / mad

    # NaN inputs stay NaN in modified_z, so exclude them from flagging
    anomalies = np.abs(modified_z_score) > threshold
    return anomalies


def mask_zero_values(data: np.ndarray) -> np.ndarray:
    """Remove zero values from an array.

    Filters out all zero values (0.0) from the input numpy array, returning only
    non-zero elements. This function is commonly used to clean data before outlier
    detection or statistical calculations.

    Args:
        data (np.ndarray): Input array of numerical data.

    Returns:
        np.ndarray: Array with zero values removed, preserving original order.

    Raises:
        TypeError: If input is not a numpy array.

    Examples:
        >>> mask_zero_values(np.array([1, 0, 2, 0, 3]))
        array([1, 2, 3])
        >>> mask_zero_values(np.array([0.0, 1.5, 0.0, 2.5]))
        array([1.5, 2.5])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    non_zero_mask = data != 0.0
    return data[non_zero_mask]


def get_completeness(stream_or_trace: Stream | Trace) -> float:
    """Check daily data completeness.

    Computes the ratio of valid (non-zero, non-NaN) samples to the expected
    total for a full 24-hour day at the stream's sampling rate.

    Args:
        stream_or_trace (Stream | Trace): ObsPy Stream or Trace object.

    Returns:
        float: Daily data completeness as a fraction in [0.0, 1.0].

    Raises:
        TypeError: If the input is not a Stream or Trace object.

    Examples:
        >>> from obspy import read
        >>> stream = read()
        >>> completeness = get_completeness(stream)
        >>> print(f"Completeness: {completeness:.2%}")
    """
    if isinstance(stream_or_trace, Stream):
        stream_or_trace = stream_or_trace.merge(fill_value=np.nan)
        trace: Trace = stream_or_trace[0]
    elif isinstance(stream_or_trace, Trace):
        trace: Trace = stream_or_trace
    else:
        raise TypeError("Input must be a Stream or Trace")

    sampling_rate = trace.stats.sampling_rate
    total_samples = sampling_rate * 60 * 60 * 24
    n_samples = count_valid_values(trace.data)
    return float(n_samples / total_samples)


def count_valid_values(data: np.ndarray) -> int:
    """Count elements that are neither zero nor NaN.

    Counts the number of valid samples in an array by excluding both
    zero values (which typically indicate dead or missing seismic channels)
    and NaN values (which indicate gaps or padding).

    Args:
        data (np.ndarray): Input array of numerical data.

    Returns:
        int: Number of elements that are not zero and not NaN.

    Raises:
        TypeError: If input is not a numpy array.

    Examples:
        >>> count_valid_values(np.array([1.0, 0.0, np.nan, 2.0]))
        2
        >>> count_valid_values(np.array([0.0, np.nan]))
        0
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    return int(np.sum((data != 0.0) & ~np.isnan(data)))


def filter_nans(data: np.ndarray) -> np.ndarray:
    """Remove NaN values from an array.

    Args:
        data (np.ndarray): Input array that may contain NaN values.

    Returns:
        np.ndarray: Array with all NaN values removed.

    Note:
        Used internally by ``detect_maximum_outlier`` and ``remove_outliers``.
    """
    return data[~np.isnan(data)]


def detect_maximum_outlier(
    data: np.ndarray, outlier_threshold: float = 3.0
) -> tuple[bool, int | float, float]:
    """Detect if maximum value in array is an outlier using z-score method.

    Uses z-score ((X - μ) / σ) to determine if the maximum value (by absolute value)
    in the array is statistically an outlier. A value is considered an outlier if its
    z-score exceeds the threshold (default 3.0, equivalent to 3 standard deviations).
    NaN values are automatically filtered before detection.

    Args:
        data (np.ndarray): Array of numerical data.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0 (3 standard deviations).

    Returns:
        tuple[bool, int | float, float]:
            - is_outlier (bool): True if maximum value is an outlier.
            - outlier_index (int | float): Index of the maximum value, or np.nan if no outlier.
            - outlier_value (float): Maximum value, or np.nan if no outlier.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If array is empty or outlier_threshold is not positive.

    Examples:
        >>> detect_maximum_outlier(np.array([1, 2, 3, 100]))  # 100 is outlier
        (True, 3, 100.0)
        >>> detect_maximum_outlier(np.array([1, 2, 3, 4]))  # No outlier
        (False, nan, nan)
        >>> detect_maximum_outlier(np.array([5, 5, 5, 5]))  # Identical values
        (False, nan, nan)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if len(data) == 0:
        raise ValueError("Cannot detect outliers in empty array")

    if outlier_threshold <= 0:
        raise ValueError("Outlier threshold must be positive")

    # Handle NaN values
    if np.any(np.isnan(data)):
        data = filter_nans(data)
        if len(data) == 0:
            return False, np.nan, np.nan

    outlier_index = np.argmax(np.abs(data))
    outlier_value = data[outlier_index]

    # If all values are identical, no outlier exists
    std = np.std(data)
    if std == 0:
        return False, np.nan, np.nan

    # Calculate z-score: Z = (X - μ) / σ
    mean = np.mean(data)
    z_score = abs((outlier_value - mean) / std)

    # Check if z-score exceeds threshold
    if z_score > outlier_threshold:
        return True, int(outlier_index), float(outlier_value)

    return False, np.nan, np.nan


def remove_maximum_outlier(
    data: np.ndarray, mask_zero_value: bool = True, outlier_threshold: float = 3.0
) -> np.ndarray:
    """Remove single maximum outlier from array using z-score method.

    Detects if the maximum value (by absolute value) is an outlier and removes it.
    This function removes at most one outlier per call. To remove all outliers,
    use remove_outliers() instead. Optionally masks zero values before detection.

    Args:
        data (np.ndarray): Input array of numerical data.
        mask_zero_value (bool, optional): If True, remove zero values before processing.
            Defaults to True.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0.

    Returns:
        np.ndarray: Array with maximum outlier removed (if detected), or original array
            if no outlier found.

    Raises:
        TypeError: If input is not a numpy array.

    Examples:
        >>> remove_maximum_outlier(np.array([1, 2, 3, 100]))
        array([1, 2, 3])
        >>> remove_maximum_outlier(np.array([0, 1, 2, 3]), mask_zero_value=False)
        array([0, 1, 2, 3])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Make a copy to avoid modifying the original
    data = data.copy()

    # Optionally mask zero values
    if mask_zero_value:
        data = mask_zero_values(data)

    # Return empty array if no data left
    if len(data) == 0:
        return np.array([])

    # Detect and remove maximum outlier
    try:
        is_outlier, outlier_index, _ = detect_maximum_outlier(
            data, outlier_threshold=outlier_threshold
        )

        if is_outlier and not np.isnan(outlier_index):
            data = np.delete(data, int(outlier_index))
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not detect outlier: {e}")
        return data

    return data


def remove_outliers(
    data: np.ndarray,
    outlier_threshold: float = 3.5,
    mask_zero_value: bool = True,
    return_outliers: bool = False,
) -> np.ndarray:
    """Remove all outliers from array based on z-score threshold.

    Removes all values whose z-score exceeds the threshold in a single pass.
    Unlike remove_maximum_outlier which removes only one value, this function
    removes all outliers simultaneously. NaN values are automatically filtered.

    Args:
        data (np.ndarray): Input array of numerical data.
        outlier_threshold (float, optional): Z-score threshold in standard deviations.
            Defaults to 3.5 (3σ).
        mask_zero_value (bool, optional): If True, remove zero values before processing.
            Defaults to True.
        return_outliers (bool, optional): If True, return outliers instead of filtered data.
            Defaults to False.

    Returns:
        np.ndarray: Array with outliers removed, or array of outliers if return_outliers=True.
            Returns empty array if all values are identical (std=0).

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If outlier_threshold is not positive.

    Examples:
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]))
        array([1, 2, 3])
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]), return_outliers=True)
        array([100, 200])
        >>> remove_outliers(np.array([1, 2, 3, 4]), outlier_threshold=2.0)
        array([1, 2, 3, 4])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if outlier_threshold <= 0:
        raise ValueError("Outlier threshold must be positive")

    # Optionally mask zero values
    if mask_zero_value:
        data = mask_zero_values(data)

    # Return empty array if no data left
    if len(data) == 0:
        return np.array([])

    # Handle NaN values
    if np.any(np.isnan(data)):
        data = filter_nans(data)
        if len(data) == 0:
            return np.array([])

    # Calculate mean and standard deviation
    mean = np.nanmean(data)
    std = np.nanstd(data)

    # If all values are identical, no outliers exist
    if std == 0:
        return np.array([]) if return_outliers else data

    # Calculate z-scores: Z = |X - μ| / σ
    z_scores = np.abs((data - mean) / std)

    # Create mask for non-outliers (z-score <= threshold)
    non_outlier_mask = z_scores <= outlier_threshold

    # Get filtered data and outliers
    filtered_data = data[non_outlier_mask]
    outliers = data[~non_outlier_mask]

    if return_outliers:
        return outliers

    return filtered_data
