# TODO

> **Note:** The HTTP API layer has been moved to a separate project (`eruption-forecast-api`) and is maintained independently. This file tracks work for the core `eruption-forecast` package only.

## Table of Contents

- [Completed](#completed)
- [High Priority](#high-priority)
- [Medium Priority](#medium-priority)
- [Low Priority](#low-priority)
- [Model-Specific Improvements](#model-specific-improvements)

---

## Completed

- [x] Add evaluation of the random forest model
- [x] Fix grammar and spelling issues in docstrings
- [x] Add examples to class docstrings (ForecastModel, ClassifierModel, TrainModel, FeaturesBuilder)
- [x] Fix deprecated `max_features="auto"` parameter (sklearn 1.4+)
- [x] Update SUMMARY.md with ML analysis and recommendations
- [x] Fix and improve docstrings in FeaturesBuilder, TremorMatrixBuilder, ForecastModel, and TrainModel
- [x] **Fix critical data leakage in TrainModel._train()** - Completely rewrote workflow to: (1) split data first, (2) resample only training data, (3) select features only on training data, (4) train classifier with GridSearchCV, (5) evaluate on held-out test set. Now saves trained models, per-seed metrics (JSON), and aggregated statistics (CSV).
- [x] **Make TrainModel classifier dynamic** - Integrated ClassifierModel to support all classifier types (RF, GB, SVM, LR, NN, DT, KNN, NB, Voting) with configurable CV strategies (shuffle, stratified, timeseries). Users can now select classifier and CV strategy during TrainModel initialization.
- [x] **Fix pyrefly type errors in ModelEvaluator** - Added Protocol types for type safety, explicit type annotations with proper ignore comments, runtime validation checks, and improved type annotations across all methods.
- [x] **Enhanced Feature Selection** - Created new FeatureSelector class with three methods: (1) tsfresh statistical selection, (2) RandomForest permutation importance, (3) combined two-stage (tsfresh → RandomForest). Provides comprehensive feature selection with both p-values and importance scores. Includes comparison script for empirical evaluation.
- [x] **Fix docstrings (2026-02-11)** - Fixed spelling errors (`Extacted`, `SKipp`, `WIll`, `laad`, `preditcted`, `defaut`, `shanon`, `paramaters`, `BE CAREFULL`). Improved class-level docstrings for `CalculateTremor`, `ClassifierModel.set_random_state`, `FeatureSelector` methods, `TremorMatrixBuilder`, and `utils.to_series`. Rewrote README.md and updated SUMMARY.md.
- [x] **Full-dataset training + ModelPredictor (2026-02-12)** - Added `TrainModel._fit()` and `fit()` for training on the full dataset without train/test split. Extended `ModelEvaluator` with `selected_features` param and `from_files()` classmethod. Created `ModelPredictor` class for evaluating trained models against future feature sets with multi-seed aggregation.
- [x] **ModelPredictor multi-model consensus (2026-02-13)** - Refactored `trained_models_csv` to `trained_models: str | dict[str, str]`. Multi-model `predict_proba()` outputs per-classifier columns + `consensus_*` columns (mean probability, inter-model uncertainty, classifier agreement). `predict()` adds `classifier` column. New `_compute_model_proba()` helper. Plot shows per-classifier dashed lines + consensus solid line.
- [x] **Imbalance-aware classifier improvements + richer evaluation (2026-02-13)** - Fixed `_train()` skip-logic bug (spurious `or not save_features` condition). Expanded RF/NN/LR/XGB hyperparameter grids; removed hardcoded `scale_pos_weight=1` from XGB (now tuned via grid). Added `class_weight`/`n_jobs` params to `ClassifierModel`. Added `optimize_threshold()`, `plot_threshold_analysis()`, `plot_feature_importance()`, `plot_calibration()`, `plot_prediction_distribution()` to `ModelEvaluator`; updated `get_metrics()` with optimal-threshold fields and `plot_all()` to include new plots.
- [x] **Add root_dir parameter (2026-02-13)** - Added `resolve_output_dir()` helper to `utils.py` and `root_dir` parameter to all standalone classes (`CalculateTremor`, `LabelBuilder`, `FeaturesBuilder`, `TremorMatrixBuilder`, `ModelTrainer`, `ModelEvaluator`) and to `ForecastModel`. All relative output paths now resolve against `root_dir` instead of `os.getcwd()`, making outputs stable regardless of the working directory. Fully backward-compatible.
- [x] **Refactor model_trainer.py (2026-02-13)** - Fixed string formatting bug in `update_grid_params` log message, fixed typo in comment, removed incorrect docstring example. Extracted `_setup_grid_search()`, `_run_jobs()`, and `_save_models_registry()` helpers to eliminate code duplication. Unified skip logic by returning `can_skip` from `_generate_filepaths()` instead of a side-effect. Renamed: `train()` → `train_and_evaluate()`, `_train()` → `_run_train_and_evaluate()`, `fit()` → `train()`, `_fit()` → `_run_train()`. Updated all call sites in `forecast_model.py`, `model_predictor.py`, and tests.
- [x] **Refactor output directory structure + docstring fixes (2026-02-15)** - Restructured `ModelTrainer` output into `evaluations/` and `predictions/` subdirectories organised by `{classifier-slug}/{cv-slug}/`. Added `get_classifier_properties()` and `update_directories()` methods. Added `with_evaluation` and `grid_params` params to `ForecastModel.train()`. Fixed `get_metrics()` parameter rename (`classifier` → `classifier_model`). Added `slugify_class_name()` to `utils.py`. Fixed docstring typos (`pd.DataFRame`, `SLugify`, `Trainig`). Updated README output structure, path examples, and ForecastModel description.
- [x] **Code quality DRY + clarity refactoring (2026-02-25)** — Extracted `BaseDataContainer(ABC)` and `SeismicDataSource(ABC)` base classes; added `_filter_nans()`, `validate_random_state()`, `load_labels_from_csv()` utilities; replaced `ClassifierModel` if-elif chains with `_GRID_REGISTRY` dict + `_build_model()` factory; split `_run_train_and_evaluate()` into three focused helpers; added `nature_figure()` context manager for plot boilerplate; extracted `parse_label_filename()` in `LabelData`; consolidated `os.makedirs` into `create_directories()`; renamed ambiguous `_cw`, `df`, `minimum_sample_acquired` variables for clarity.
- [x] **SeedEnsemble — merge 500 seed models into one file (2026-02-25)** — Created `SeedEnsemble` class (sklearn `BaseEstimator` + `ClassifierMixin`) that bundles all seed estimators + feature lists for a single classifier into one serialisable `.pkl`. Added `merge_seed_models()` and `merge_all_classifiers()` to `utils/ml.py`. Added `merge_models()` and `merge_classifier_models()` thin wrappers to `ModelTrainer`. Extended `ModelPredictor` to accept merged `.pkl` paths (single `SeedEnsemble` or multi-classifier `dict[str, SeedEnsemble]`) in addition to the existing CSV-based registry path.

## High Priority

- [x] Add Gradient Boosting classifier (GradientBoostingClassifier or XGBoost)
- [x] Implement TimeSeriesSplit for proper temporal cross-validation
- [x] Add model persistence with joblib (save/load trained models) - via ModelEvaluator
- [x] Expand grid parameters for neural network (hidden_layer_sizes too limited)

## Medium Priority

- [x] Add VotingClassifier ensemble combining top models
- [x] Add precision-recall curves and ROC-AUC evaluation metrics (via ModelEvaluator)
- [x] Add detailed evaluation metrics with export and plotting (ModelEvaluator class)
- [x] Implement enhanced feature selection (tsfresh + RandomForest + combined)
- [ ] Implement probability calibration (CalibratedClassifierCV)
- [ ] Add StandardScaler to pipeline for SVM/KNN/NN/LR (use `sklearn.pipeline.Pipeline` to prevent data leakage)
- [ ] Fix `value_multiplier` ignored when `remove_outlier_method=None` in `window.py` (logic issue from REVIEW.md)
- [ ] Fix `set_random_state()` not invalidating cached `_model` in `classifier_model.py` (logic issue from REVIEW.md)
- [ ] Guard `shutil.rmtree` with `os.path.exists` check in `calculate_tremor.py` (logic issue from REVIEW.md)
- [ ] Fix `source="sds"` with `sds_dir=None` silently skipping in `forecast_model.py` (logic issue from REVIEW.md)
- [ ] Remove `sleep(3)` in `ForecastModel.train()` (quality issue from REVIEW.md)
- [ ] Rename `shanon`→`shannon` throughout (breaking rename — defer to next major version)

## Low Priority

- [ ] Expand integration tests
- [ ] Add end-to-end workflow tests
- [ ] Create user documentation
- [ ] Add example Jupyter notebooks
- [ ] Performance benchmarking
- [ ] **GPU acceleration** — Enable XGBoost GPU (`device="cuda"` in `classifier_model.py:444`); consider RAPIDS cuML (WSL2/Linux only) for RF/SVM/LR/KNN GPU support. XGBoost change is one line and works natively on Windows with NVIDIA GPU + CUDA drivers. tsfresh and GridSearchCV are CPU-bound and not GPU-friendly.
- [ ] Explore LSTM/Transformer for sequential patterns
- [ ] Add SHAP values for model explainability

## Model-Specific Improvements

### Random Forest
- [x] Expand n_estimators grid to [50, 100, 200]
- [ ] Add max_depth options [10, 15, None] for complex patterns
- [x] Add min_samples_split and min_samples_leaf parameters

### Neural Network
- [x] Expand hidden_layer_sizes to multi-layer architectures
- [ ] Add early_stopping and validation_fraction
- [x] Add learning_rate_init parameter options

### Logistic Regression
- [x] Add solver parameter for l1/elasticnet support
- [x] Add l1_ratio parameter for elasticnet

### Naive Bayes
- [ ] Expand var_smoothing grid to [1e-9, 1e-8, 1e-7, 1e-6] (also fixes the no-op grid from REVIEW.md)
