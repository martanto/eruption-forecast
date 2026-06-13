`EvaluationModel(BaseModel)` measures a fitted ensemble against ground truth — without ever re-fitting.

Two modes (dispatched on `model.kind`):
- **Training reuse** — `y_true` is `TrainingModel.labels`; features/ensemble taken from the training stage; `eruption_dates` optional.
- **Prediction reuse** — `eruption_dates` required so a fresh `LabelBuilder` can stamp truth onto the forecast window grid; `y_true` is then joined onto `PredictionModel.labels` by window id and persisted at `evaluation/prediction/labels/y_true.csv`.

`evaluate()` builds a `MetricsEnsemble` once and caches it on `self.MetricsEnsemble`; `compute()` materialises `(n_samples, n_seeds)` `y_proba`/`y_pred` matrices per classifier under `classifiers/{clf}/predictions/`. Optional `plot_aggregate=True` and `plot_per_seed=True` render the corresponding figure dispatchers. `compare()` builds a `ClassifierComparator` (lazily) and writes ranking tables + cross-classifier plots under `evaluation/{kind}/comparison/`.
