import numpy as np
import pandas as pd
from typing import Optional


def detect_outliers(
    data: np.ndarray, outlier_threshold: Optional[float] = 3.0
) -> tuple[bool, int, float]:
    """Detect outliers in an array and return an array with outliers
    using z-score ((X - μ) / σ)

    Args:
        data (np.ndarray): Array of data from trace
        outlier_threshold (float, optional): Degree of outliers. Defaults to 0.5.

    Returns:
        tuple[bool, int, float]:
            outlier (bool) : true if outlier is detected, false otherwise.
            outlier_index : Index of the outlier
            outlier_value : Value of the outlier
    """
    if isinstance(data, pd.Series):
        data = data.values

    outlier_index = np.argmax(data)
    outlier_value = data[outlier_index]

    # Z-score = (X - μ) / σ
    try:
        z_score = (outlier_value - np.mean(data)) / np.std(data)
    except ZeroDivisionError:
        return False, np.nan, np.nan

    if z_score > outlier_threshold:
        return True, outlier_index, float(outlier_value)

    return False, np.nan, np.nan
