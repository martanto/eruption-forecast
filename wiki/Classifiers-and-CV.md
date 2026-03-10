# Classifiers and Cross-Validation

## Supported Classifiers

Pass any `classifier` key to `ModelTrainer` or `ForecastModel.train()`.

| Key | Full Name | Imbalance Handling | Notes |
|-----|-----------|--------------------|-------|
| `rf` | Random Forest | `class_weight="balanced"` | Robust default; good all-round performance |
| `gb` | Gradient Boosting | None (natural) | Handles imbalance natively via boosting weights |
| `xgb` | XGBoost | `scale_pos_weight` grid search | Excellent for imbalanced data; most tunable; **GPU-capable** (`use_gpu=True`) |
| `svm` | Support Vector Machine | `class_weight="balanced"` | Effective in high-dimensional feature spaces |
| `lr` | Logistic Regression | `class_weight="balanced"` | Fast and interpretable; good baseline |
| `nn` | Neural Network (MLP) | None | Captures complex non-linear patterns |
| `dt` | Decision Tree | `class_weight="balanced"` | Interpretable single-tree baseline |
| `knn` | K-Nearest Neighbors | None | Simple distance-based baseline |
| `nb` | Gaussian Naive Bayes | None | Very fast probabilistic baseline |
| `voting` | Soft Voting Ensemble | Combined | Combines RF + XGBoost with probability averaging; **GPU-capable** (XGBoost component only) |
| `lite-rf` | Random Forest (lite grid) | `class_weight="balanced"` | Smaller hyperparameter grid for faster training |

### Training multiple classifiers

Pass a `list[str]` or comma-separated string. Each classifier trains sequentially; all resulting model registries are stored in `ForecastModel.trained_models` for multi-model consensus forecasting.

```python
fm.train(
    classifier=["rf", "xgb", "gb"],
    cv_strategy="stratified",
    total_seed=500,
    with_evaluation=False,
)

print(fm.trained_models)
# {
#   "RandomForestClassifier": "output/.../trained_model_rf_*.csv",
#   "XGBClassifier":          "output/.../trained_model_xgb_*.csv",
#   "GradientBoostingClassifier": "output/.../trained_model_gb_*.csv",
# }
```

---

## Hyperparameter Grids

Each classifier ships with a default `GridSearchCV` grid. The best hyperparameters are selected independently per seed.

### Random Forest (`rf`)

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
```

### Gradient Boosting (`gb`)

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
```

### XGBoost (`xgb`)

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "scale_pos_weight": [1, 5, 10, 15],
}
```

> `scale_pos_weight` controls extra weight given to eruption (positive) samples. Higher values increase sensitivity at the cost of more false positives.

### Voting Ensemble (`voting`)

```python
{
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, None],
    "xgb__n_estimators": [50, 100],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__max_depth": [5, 7],
}
```

Combines RF and XGBoost with soft voting (probability averaging).

### Overriding the default grid

```python
from eruption_forecast.model.classifier_model import ClassifierModel

clf = ClassifierModel("xgb", random_state=42)
clf.grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "scale_pos_weight": [1, 5, 10, 20],
}
```

---

## Cross-Validation Strategies

| Key | scikit-learn Class | Best For |
|-----|--------------------|----------|
| `shuffle` | `ShuffleSplit` | Random splits without stratification — fast baseline |
| `stratified` | `StratifiedKFold` | Preserves class ratio across all folds |
| `shuffle-stratified` | `StratifiedShuffleSplit` | Randomized stratified folds — **default** |
| `timeseries` | `TimeSeriesSplit` | Temporal data — strict no-future-leakage ordering |

Set via `cv_strategy` in `ModelTrainer` or `ForecastModel.train()`. Number of folds is controlled by `cv_splits` (default: 5).

### When to use each

- **`shuffle`** — fastest; no stratification; good for quick experiments when class balance is not critical.
- **`stratified`** — use when class imbalance is severe and every fold must represent both classes fairly across a fixed split structure.
- **`shuffle-stratified`** — **recommended default**; combines random splits with class stratification via `StratifiedShuffleSplit`; each seed gets a different random stratified split, which works well for multi-seed training.
- **`timeseries`** — use when temporal ordering matters and you want to strictly prevent the model from seeing future data during cross-validation. Note: this strategy does not stratify, so very small minority classes may be absent from some folds.

---

## Comparing Classifiers

After training multiple classifiers with `with_evaluation=True`, compare their aggregate metrics:

```python
import pandas as pd

base = "output/trainings/evaluations"
suffix = "rs-0_ts-500_top-20"
classifiers = {
    "rf":  f"{base}/random-forest-classifier/stratified-shuffle-split/all_metrics_RandomForestClassifier-StratifiedShuffleSplit_{suffix}.csv",
    "xgb": f"{base}/xgb-classifier/stratified-shuffle-split/all_metrics_XGBClassifier-StratifiedShuffleSplit_{suffix}.csv",
    "gb":  f"{base}/gradient-boosting-classifier/stratified-shuffle-split/all_metrics_GradientBoostingClassifier-StratifiedShuffleSplit_{suffix}.csv",
}

results = []
for name, path in classifiers.items():
    df = pd.read_csv(path)
    results.append({
        "classifier":        name,
        "mean_balanced_acc": df["balanced_accuracy"].mean(),
        "std_balanced_acc":  df["balanced_accuracy"].std(),
        "mean_f1":           df["f1_score"].mean(),
        "mean_roc_auc":      df["roc_auc"].mean(),
    })

comparison = pd.DataFrame(results).sort_values("mean_balanced_acc", ascending=False)
print(comparison.to_string(index=False))
```

Or use `plot_classifier_comparison()` for a visual heatmap — see the [Visualization](Visualization) page.
