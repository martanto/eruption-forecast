# TODO

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
- [ ] Add StandardScaler to pipeline for SVM/KNN/NN

## Low Priority

- [ ] Expand integration tests
- [ ] Add end-to-end workflow tests
- [ ] Create user documentation
- [ ] Add example Jupyter notebooks
- [ ] Performance benchmarking
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
- [ ] Expand var_smoothing grid to [1e-9, 1e-8, 1e-7, 1e-6]
