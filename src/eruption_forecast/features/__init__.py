"""Feature extraction and selection module for volcanic eruption forecasting.

This module provides tools for building tremor matrices, extracting time-series
features using tsfresh, and performing two-stage feature selection combining
statistical testing and machine learning importance.

Classes:
    TremorMatrixBuilder: Slices tremor data into windowed matrices aligned with labels.
    FeaturesBuilder: Extracts tsfresh features from tremor matrices.
    FeatureSelector: Two-stage feature selection (tsfresh + RandomForest).

Constants:
    ID_COLUMN: Column name for window identifiers.
    DATETIME_COLUMN: Column name for datetime values.
    ERUPTED_COLUMN: Column name for eruption labels.

Examples:
    >>> # Build tremor matrix
    >>> matrix_builder = TremorMatrixBuilder(tremor_df, label_df)
    >>> matrix_builder.build(select_tremor_columns=["rsam_f0", "rsam_f1"])
    >>>
    >>> # Extract features
    >>> features_builder = FeaturesBuilder(matrix_builder.df, label_df=label_df)
    >>> features = features_builder.extract_features()
    >>>
    >>> # Select features
    >>> selector = FeatureSelector(method="combined")
    >>> X_selected = selector.fit_transform(features, y_train)
"""
