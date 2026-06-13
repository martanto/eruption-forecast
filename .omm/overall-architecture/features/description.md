Feature-engineering domain. Slices tremor series into per-window samples, extracts time-series features via tsfresh, and selects the most relevant subset for downstream models.

Files:
- tremor_matrix_builder.py — `TremorMatrixBuilder`: aligns tremor rows with the label window grid, validates sample counts per window, and emits a long-form matrix keyed by `(id, datetime)`.
- features_builder.py — `FeaturesBuilder`: tsfresh feature extraction per tremor column. Training mode filters windows + saves an aligned label CSV; prediction mode disables relevance filtering.
- feature_selector.py — `FeatureSelector`: two-stage selection (tsfresh FDR → RandomForest importance). Supports `tsfresh`, `random_forest`, and `combined` strategies; produces a per-seed top-N feature list.
- constants.py — column-name constants (`ID_COLUMN`, `ERUPTED_COLUMN`, `DATETIME_COLUMN`, `DEFAULT_EXCLUDE_FEATURES`).
