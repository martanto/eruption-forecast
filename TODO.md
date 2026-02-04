# TODO

## Completed

- [x] Add evaluation of the random forest model
- [x] Fix grammar and spelling issues in docstrings
- [x] Add examples to class docstrings (ForecastModel, ClassifierModel, TrainModel, FeaturesBuilder)
- [x] Fix deprecated `max_features="auto"` parameter (sklearn 1.4+)
- [x] Update SUMMARY.md with ML analysis and recommendations

## High Priority

- [ ] Add Gradient Boosting classifier (GradientBoostingClassifier or XGBoost)
- [ ] Implement TimeSeriesSplit for proper temporal cross-validation
- [ ] Add model persistence with joblib (save/load trained models)
- [ ] Expand grid parameters for neural network (hidden_layer_sizes too limited)

## Medium Priority

- [ ] Add VotingClassifier ensemble combining top models
- [ ] Implement probability calibration (CalibratedClassifierCV)
- [ ] Add StandardScaler to pipeline for SVM/KNN/NN
- [ ] Add precision-recall curves and ROC-AUC evaluation metrics
- [ ] Implement Recursive Feature Elimination (RFE) as alternative to tsfresh selection

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
- [ ] Expand n_estimators grid to [50, 100, 200, 300]
- [ ] Add max_depth options [10, 15, None] for complex patterns
- [ ] Add min_samples_split and min_samples_leaf parameters

### Neural Network
- [ ] Expand hidden_layer_sizes to multi-layer architectures
- [ ] Add early_stopping and validation_fraction
- [ ] Add learning_rate parameter options

### Logistic Regression
- [ ] Add solver parameter for l1/elasticnet support
- [ ] Add l1_ratio parameter for elasticnet

### Naive Bayes
- [ ] Expand var_smoothing grid to [1e-9, 1e-8, 1e-7, 1e-6]
