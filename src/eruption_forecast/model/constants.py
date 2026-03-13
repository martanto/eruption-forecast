"""Default hyperparameter grids for all supported classifiers.

This module defines ``DEFAULT_GRID_PARAMS``, a mapping from classifier
short-names to hyperparameter search grids compatible with
``sklearn.model_selection.GridSearchCV``.  Each entry covers a curated
range of values that balance search breadth with computational cost and
are tuned for imbalanced volcanic-eruption datasets.
"""

from typing import Any


CLASSIFIERS: list[str] = [
    "svm",
    "knn",
    "dt",
    "rf",
    "gb",
    "xgb",
    "nn",
    "nb",
    "lr",
    "voting",
    "lite-rf",
]

DEFAULT_GRID_PARAMS: dict[str, dict[str, Any] | list[dict[str, Any]]] = {
    "svm": [
        {"C": [0.001, 0.01, 0.1, 1, 10], "kernel": ["poly"], "degree": [2, 3], "decision_function_shape": ["ovr"]},
        {"C": [0.001, 0.01, 0.1, 1, 10], "kernel": ["rbf"], "decision_function_shape": ["ovr"]},
        {"C": [0.001, 0.01, 0.1, 1, 10], "kernel": ["linear"], "decision_function_shape": ["ovr"]},
    ],
    "knn": {
        "n_neighbors": [3, 6, 12, 24],
        "weights": ["uniform", "distance"],
        "p": [1, 2, 3],
    },
    "dt": {
        "max_depth": [3, 5, 7],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
    },
    "rf": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "gb": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "xgb": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
        "scale_pos_weight": [1],  # RandomUnderSampler already balances classes
    },
    "nn": {
        "activation": ["logistic", "tanh", "relu"],  # "identity" dropped — linear, rarely useful
        "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
        "learning_rate_init": [0.001, 0.01],
    },
    "nb": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]},
    "lr": [
        {"penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
        {"penalty": ["l1"], "C": [0.001, 0.01, 0.1, 1, 10], "solver": ["saga"]},
        {
            "penalty": ["elasticnet"],
            "C": [0.001, 0.01, 0.1, 1, 10],
            "solver": ["saga"],
            "l1_ratio": [0.15, 0.5, 0.85],
        },
    ],
    "voting": {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [10, None],
        "xgb__n_estimators": [50, 100],
        "xgb__learning_rate": [0.05, 0.1],
        "xgb__max_depth": [5, 7],
    },
    "lite-rf": {
        "n_estimators": [10, 30, 100],
        "max_depth": [3, 5, 7],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
    },
}
