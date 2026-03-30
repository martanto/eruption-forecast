"""Feature extraction and selection package for volcanic eruption forecasting.

Transforms windowed tremor time-series into machine-learning-ready
feature matrices using tsfresh automated feature extraction followed by
two-stage feature selection.  It covers three distinct steps in the pipeline:
building the windowed tremor matrix, extracting features, and selecting the most
informative subset.

Key classes:
    - ``TremorMatrixBuilder``: Slices a tremor DataFrame into fixed-size windows
      aligned with label windows and concatenates them into a matrix with ``id``,
      ``datetime``, and tremor metric columns ready for tsfresh input.
    - ``FeaturesBuilder``: Runs tsfresh feature extraction on each tremor column
      independently. Operates in training mode (with labels, relevance filtering)
      or prediction mode (all features, no filtering). Saves extracted features
      and aligned label CSVs to the ``features/`` output directory.
    - ``FeatureSelector``: Two-stage selection combining tsfresh FDR-controlled
      statistical testing (Stage 1) with RandomForest permutation importance
      (Stage 2). Supports methods ``"tsfresh"``, ``"random_forest"``, and
      ``"combined"``.
"""
