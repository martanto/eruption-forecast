"""Benchmark utilities for comparing :class:`FeatureSelector` methods."""

import os
import time
from typing import Literal
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import resample
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


FeatureMethod = Literal["tsfresh", "random_forest"]

DEFAULT_TREMOR_COLUMNS: list[str] = [
    "rsam_f2",
    "rsam_f3",
    "rsam_f4",
    "dsar_f3-f4",
    "entropy",
]
DEFAULT_EXCLUDE_FEATURES: list[str] = [
    "agg_linear_trend",
    "linear_trend_timewise",
    "length",
    "has_duplicate_max",
    "has_duplicate_min",
    "has_duplicate",
]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _build_training_model(
    tremor_csv: str,
    start_date: str,
    end_date: str,
    eruption_dates: list[str],
    select_tremor_columns: list[str],
    exclude_features: list[str],
    top_n_features: int,
    output_dir: str | None,
    root_dir: str | None,
    n_jobs: int,
    n_grids: int,
    verbose: bool,
) -> TrainingModel:
    return (
        TrainingModel(
            tremor_data=tremor_csv,
            start_date=start_date,
            end_date=end_date,
            classifiers=["rf"],
            eruption_dates=eruption_dates,
            window_size=2,
            cv_strategy="shuffle-stratified",
            cv_splits=5,
            top_n_features=top_n_features,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            n_grids=n_grids,
            verbose=verbose,
        )
        .build_label(window_step=6, window_step_unit="hours", builder="standard")
        .extract_features(
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=False,
            exclude_features=exclude_features,
            minimum_completion=1.0,
        )
    )


def _run_seed(
    seed: int,
    method: FeatureMethod,
    features_df: pd.DataFrame,
    labels: pd.Series,
    top_n_features: int,
    resample_method: Literal["under", "over"] | None,
    sampling_strategy: str | float,
    n_grids: int,
    verbose: bool,
) -> tuple[float, float, list[str], float, float, float, float]:
    """Run one ``(seed, method)`` iteration and return timings, top features, and metrics."""
    features_resampled, labels_resampled = resample(
        features=features_df,
        labels=labels,
        method=resample_method,
        sampling_strategy=sampling_strategy,
        random_state=seed,
        verbose=False,
    )

    selector = FeatureSelector(
        method=method,
        random_state=seed,
        n_jobs=n_grids,
        verbose=verbose,
    )

    started = time.perf_counter()
    features_selected = selector.fit_transform(
        features_resampled, labels_resampled, top_n=top_n_features
    )
    selection_seconds = time.perf_counter() - started
    selected_features = list(features_selected.columns)

    classifier = ClassifierModel(
        classifier="rf",
        cv_strategy="shuffle-stratified",
        n_splits=5,
        verbose=False,
    ).set_random_state(seed)

    estimator = classifier.model
    cv_splitter = classifier.get_cv_splitter()

    started = time.perf_counter()
    balanced_accuracy_scores = cross_val_score(
        estimator,
        features_selected,
        labels_resampled,
        cv=cv_splitter,
        scoring="balanced_accuracy",
        n_jobs=n_grids,
    )
    roc_auc_scores = cross_val_score(
        estimator,
        features_selected,
        labels_resampled,
        cv=cv_splitter,
        scoring="roc_auc",
        n_jobs=n_grids,
    )
    train_seconds = time.perf_counter() - started

    return (
        selection_seconds,
        train_seconds,
        selected_features,
        float(np.mean(balanced_accuracy_scores)),
        float(np.std(balanced_accuracy_scores)),
        float(np.mean(roc_auc_scores)),
        float(np.std(roc_auc_scores)),
    )


def _mean_pairwise_jaccard(
    selected_per_method: dict[str, dict[int, list[str]]],
    methods: tuple[FeatureMethod, ...],
) -> float:
    """Mean Jaccard overlap across every method pair and every shared seed.

    Returns 0.0 when fewer than two methods are present (nothing to compare).
    """
    if len(methods) < 2:
        return 0.0

    pair_overlaps: list[float] = []
    for left, right in combinations(methods, 2):
        seeds_present = sorted(
            set(selected_per_method[left]) & set(selected_per_method[right])
        )
        for seed in seeds_present:
            pair_overlaps.append(
                _jaccard(
                    set(selected_per_method[left][seed]),
                    set(selected_per_method[right][seed]),
                )
            )

    return float(np.mean(pair_overlaps)) if pair_overlaps else 0.0


def _write_outputs(
    bench_dir: str,
    methods: tuple[FeatureMethod, ...],
    runtime_rows: list[dict],
    selected_per_method: dict[str, dict[int, list[str]]],
    metrics_per_method: dict[str, list[dict]],
) -> tuple[pd.DataFrame, float]:
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(os.path.join(bench_dir, "runtime.csv"), index=False)

    for method in methods:
        sel_rows = [
            {"seed": seed, "rank": rank, "feature": feature_name}
            for seed, feature_names in sorted(selected_per_method[method].items())
            for rank, feature_name in enumerate(feature_names)
        ]
        pd.DataFrame(sel_rows, columns=["seed", "rank", "feature"]).to_csv(
            os.path.join(bench_dir, f"selected_features_{method}.csv"), index=False
        )

        pd.DataFrame(metrics_per_method[method]).to_csv(
            os.path.join(bench_dir, f"metrics_{method}.csv"), index=False
        )

        flat_features = [
            feature_name
            for feature_names in selected_per_method[method].values()
            for feature_name in feature_names
        ]
        frequency = pd.Series(flat_features).value_counts()
        frequency.name = "count"
        frequency.index.name = "feature"
        frequency.to_csv(os.path.join(bench_dir, f"feature_frequency_{method}.csv"))

    summary_rows: list[dict] = []
    for method in methods:
        method_runtime = runtime_df[runtime_df["method"] == method]
        method_metrics = pd.DataFrame(metrics_per_method[method])
        summary_rows.append(
            {
                "method": method,
                "selection_mean_s": method_runtime["selection_seconds"].mean(),
                "selection_std_s": method_runtime["selection_seconds"].std(),
                "train_mean_s": method_runtime["train_seconds"].mean(),
                "balanced_accuracy_mean": method_metrics[
                    "balanced_accuracy_mean"
                ].mean(),
                "balanced_accuracy_std": method_metrics[
                    "balanced_accuracy_mean"
                ].std(),
                "roc_auc_mean": method_metrics["roc_auc_mean"].mean(),
                "roc_auc_std": method_metrics["roc_auc_mean"].std(),
            }
        )

    mean_overlap = _mean_pairwise_jaccard(selected_per_method, methods)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["mean_jaccard_vs_other"] = mean_overlap
    summary_df.to_csv(os.path.join(bench_dir, "summary.csv"), index=False)

    return summary_df, mean_overlap


def _render_plots(
    bench_dir: str,
    methods: tuple[FeatureMethod, ...],
    runtime_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_df.boxplot(column="selection_seconds", by="method", ax=ax)
    ax.set_title("FeatureSelector.fit_transform runtime per seed")
    ax.set_ylabel("seconds")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(bench_dir, "runtime_box.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(methods))
    width = 0.35
    balanced_accuracy_means = summary_df["balanced_accuracy_mean"].to_numpy()
    balanced_accuracy_stds = summary_df["balanced_accuracy_std"].to_numpy()
    roc_auc_means = summary_df["roc_auc_mean"].to_numpy()
    roc_auc_stds = summary_df["roc_auc_std"].to_numpy()
    ax.bar(
        xs - width / 2,
        balanced_accuracy_means,
        width,
        yerr=balanced_accuracy_stds,
        label="balanced_accuracy",
        capsize=4,
    )
    ax.bar(
        xs + width / 2,
        roc_auc_means,
        width,
        yerr=roc_auc_stds,
        label="roc_auc",
        capsize=4,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(list(methods))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Downstream RF metrics per FeatureSelector method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(bench_dir, "metrics_compare.png"), dpi=150)
    plt.close(fig)


def benchmark_feature_selection(
    tremor_csv: str,
    training_start_date: str,
    training_end_date: str,
    eruption_dates: list[str],
    select_tremor_columns: list[str] | None = None,
    exclude_features: list[str] | None = None,
    output_dir: str | None = None,
    root_dir: str | None = None,
    seeds: int = 10,
    top_n_features: int = 20,
    n_jobs: int = 1,
    n_grids: int = 1,
    methods: tuple[FeatureMethod, ...] = ("tsfresh", "random_forest"),
    resample_method: Literal["under", "over"] | None = "under",
    sampling_strategy: str | float = 0.75,
    verbose: bool = False,
) -> None:
    """Benchmark :class:`FeatureSelector` methods side-by-side on one tremor dataset.

    Builds labels and the tsfresh feature matrix once via :class:`TrainingModel`,
    then for every ``(seed, method)`` pair re-fits a fresh
    :class:`FeatureSelector`, times its ``fit_transform`` call, records the
    top-N feature set, and evaluates a downstream RandomForest with
    :func:`sklearn.model_selection.cross_val_score` for ``balanced_accuracy``
    and ``roc_auc``. Aggregates everything into CSVs + comparison plots under
    ``{output_dir}/{nslc}/benchmarks/feature-selection/``.

    Args:
        tremor_csv (str): Path to a pre-computed tremor CSV (output of
            :class:`CalculateTremor`).
        training_start_date (str): Start of the training window
            (``YYYY-MM-DD``).
        training_end_date (str): End of the training window (``YYYY-MM-DD``).
        eruption_dates (list[str]): Eruption dates used to label positives.
        select_tremor_columns (list[str] | None): Tremor columns to use.
            Defaults to ``DEFAULT_TREMOR_COLUMNS``.
        exclude_features (list[str] | None): tsfresh feature names to drop.
            Defaults to ``DEFAULT_EXCLUDE_FEATURES``.
        output_dir (str | None): Station output sub-directory under
            ``root_dir``. Forwarded to :class:`TrainingModel`.
        root_dir (str | None): Project root for output resolution.
        seeds (int): Number of random seeds to benchmark. Defaults to ``10``.
        top_n_features (int): Top-N feature cap passed to
            :meth:`FeatureSelector.fit_transform`. Defaults to ``20``.
        n_jobs (int): Outer worker count forwarded to :class:`TrainingModel`.
            Defaults to ``1``.
        n_grids (int): Inner parallelism for :class:`FeatureSelector` and
            ``cross_val_score``. Defaults to ``1``.
        methods (tuple[FeatureMethod, ...]): Selection methods to compare.
            Defaults to ``("tsfresh", "random_forest")``.
        resample_method (Literal["under", "over"] | None): Resampling
            strategy applied before selection. Defaults to ``"under"``.
        sampling_strategy (str | float): Target ratio forwarded to the
            resampler. Defaults to ``0.75``.
        verbose (bool): Forwarded to :class:`TrainingModel` and
            :class:`FeatureSelector`. Defaults to ``False``.

    Raises:
        FileNotFoundError: If ``tremor_csv`` does not exist on disk.

    Example:
        >>> from eruption_forecast.utils.benchmark import benchmark_feature_selection
        >>> benchmark_feature_selection(
        ...     tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_VG.OJN.00.EHZ.csv",
        ...     training_start_date="2025-01-01",
        ...     training_end_date="2025-07-26",
        ...     eruption_dates=["2025-03-20", "2025-04-10"],
        ...     output_dir="VG.OJN.00.EHZ",
        ...     root_dir="D:/Projects/eruption-forecast/output",
        ...     seeds=10,
        ...     n_jobs=4,
        ...     n_grids=2,
        ... )
    """
    if not os.path.isfile(tremor_csv):
        raise FileNotFoundError(
            f"Tremor CSV not found: {tremor_csv}\n"
            "Run ForecastModel.calculate(...) (or training-ojn.py prerequisites) "
            "to build the tremor matrix first."
        )

    select_tremor_columns = (
        select_tremor_columns
        if select_tremor_columns is not None
        else list(DEFAULT_TREMOR_COLUMNS)
    )
    exclude_features = (
        exclude_features
        if exclude_features is not None
        else list(DEFAULT_EXCLUDE_FEATURES)
    )

    training_model = _build_training_model(
        tremor_csv=tremor_csv,
        start_date=training_start_date,
        end_date=training_end_date,
        eruption_dates=eruption_dates,
        select_tremor_columns=select_tremor_columns,
        exclude_features=exclude_features,
        top_n_features=top_n_features,
        output_dir=output_dir,
        root_dir=root_dir,
        n_jobs=n_jobs,
        n_grids=n_grids,
        verbose=verbose,
    )
    bench_dir = os.path.join(
        training_model.output_dir, "benchmarks", "feature-selection"
    )
    os.makedirs(bench_dir, exist_ok=True)

    logger.info(
        f"Loaded features: {training_model.features_df.shape} | "
        f"labels: {training_model.labels.shape}"
    )

    runtime_rows: list[dict] = []
    selected_per_method: dict[str, dict[int, list[str]]] = {
        method: {} for method in methods
    }
    metrics_per_method: dict[str, list[dict]] = {method: [] for method in methods}

    for seed in range(seeds):
        for method in methods:
            logger.info(f"Seed {seed:05d} / {method}: running benchmark...")
            (
                selection_seconds,
                train_seconds,
                selected_features,
                balanced_accuracy_mean,
                balanced_accuracy_std,
                roc_auc_mean,
                roc_auc_std,
            ) = _run_seed(
                seed=seed,
                method=method,
                features_df=training_model.features_df,
                labels=training_model.labels,
                top_n_features=top_n_features,
                resample_method=resample_method,
                sampling_strategy=sampling_strategy,
                n_grids=n_grids,
                verbose=verbose,
            )

            selected_per_method[method][seed] = selected_features
            runtime_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "selection_seconds": selection_seconds,
                    "train_seconds": train_seconds,
                    "n_selected": len(selected_features),
                }
            )
            metrics_per_method[method].append(
                {
                    "seed": seed,
                    "balanced_accuracy_mean": balanced_accuracy_mean,
                    "balanced_accuracy_std": balanced_accuracy_std,
                    "roc_auc_mean": roc_auc_mean,
                    "roc_auc_std": roc_auc_std,
                }
            )

    summary_df, mean_overlap = _write_outputs(
        bench_dir=bench_dir,
        methods=methods,
        runtime_rows=runtime_rows,
        selected_per_method=selected_per_method,
        metrics_per_method=metrics_per_method,
    )
    _render_plots(
        bench_dir=bench_dir,
        methods=methods,
        runtime_df=pd.DataFrame(runtime_rows),
        summary_df=summary_df,
    )

    logger.info("=" * 64)
    for _, row in summary_df.iterrows():
        logger.info(
            f"{row['method']:<9}: selection={row['selection_mean_s']:6.2f}s "
            f"(+/-{row['selection_std_s']:.2f}), "
            f"bal_acc={row['balanced_accuracy_mean']:.3f} "
            f"(+/-{row['balanced_accuracy_std']:.3f}), "
            f"roc_auc={row['roc_auc_mean']:.3f} "
            f"(+/-{row['roc_auc_std']:.3f})"
        )
    logger.info(
        f"mean Jaccard overlap between top-{top_n_features} sets: {mean_overlap:.3f}"
    )
    logger.info(f"Outputs written to: {bench_dir}")
