import os
import json
import warnings
from typing import Any

import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve

from eruption_forecast.logger import logger
from eruption_forecast.model.constants import LEARNING_CURVE_SCORER_MAP
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.constants import LEARNING_CURVE_SCORINGS
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.model.base_model_trainer import BaseModelTrainer


class EvaluationTrainer(BaseModelTrainer):
    """Extends BaseModelTrainer with train_and_evaluate() and its helper methods.

    Adds the full evaluation pipeline: per-seed train/test split, GridSearchCV
    training, held-out test evaluation, learning curves, SHAP, and metric
    aggregation across seeds.

    All constructor arguments are forwarded to :class:`BaseModelTrainer`.
    """

    def compute_learning_curve(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        scoring: str = "balanced_accuracy",
    ) -> dict:
        """Compute a learning curve for a single scoring metric.

        Evaluates ``estimator`` at ten linearly spaced training-set fractions
        (0.1 → 1.0) using the configured CV splitter and the given ``scoring``
        string.  Mean and standard deviation of train and validation scores are
        computed across CV folds and returned as a dict (no file I/O).

        Args:
            estimator (Any): Fitted or unfitted sklearn-compatible estimator.
            X_train (pd.DataFrame): Training features (post-selection).
            y_train (pd.Series): Training labels.
            random_state (int): Random seed used to configure the CV splitter.
            scoring (str, optional): Sklearn scoring string. Defaults to
                ``"balanced_accuracy"``.

        Returns:
            dict: Dict with keys ``train_sizes``, ``train_scores_mean``,
                ``train_scores_std``, ``test_scores_mean``, ``test_scores_std``.
        """
        # StratifiedKFold ensures every fold contains both classes, preventing
        # the single-label warning that occurs when small train_sizes + class
        # imbalance leave only one class in a fold.
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=random_state
        )

        # Force n_jobs=1: already inside a loky parallel worker (outer seed loop).
        # Suppress the single-label UserWarning: at very small training-size
        # fractions (≤20%) the model may predict only one class, causing
        # balanced_accuracy_score to warn about a single label in y_pred.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found",
                category=UserWarning,
            )
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=X_train,
                y=y_train,
                cv=cv,
                scoring=LEARNING_CURVE_SCORER_MAP.get(scoring, scoring),
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=1,
            )

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "train_scores_std": train_scores.std(axis=1).tolist(),
            "test_scores_mean": test_scores.mean(axis=1).tolist(),
            "test_scores_std": test_scores.std(axis=1).tolist(),
        }

    def _compute_all_learning_curves(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        filepath: str,
    ) -> dict:
        """Compute learning curves for all configured scoring metrics and save as JSON.

        Iterates over ``LEARNING_CURVE_SCORINGS``, calls
        :meth:`compute_learning_curve` for each scoring string, collects
        per-metric results under a ``"metrics"`` key, and writes a single JSON
        file.  ``train_sizes`` is shared across all metrics.

        Args:
            estimator (Any): Fitted or unfitted sklearn-compatible estimator.
            X_train (pd.DataFrame): Training features (post-selection).
            y_train (pd.Series): Training labels.
            random_state (int): Random seed used to configure the CV splitter.
            filepath (str): Destination path for the JSON output file.

        Returns:
            dict: Dict with keys ``random_state``, ``train_sizes``, and
                ``metrics`` (a nested dict keyed by scoring name).
        """
        metrics: dict[str, dict] = {}
        train_sizes = None
        for scoring in LEARNING_CURVE_SCORINGS:
            result = self.compute_learning_curve(
                estimator, X_train, y_train, random_state, scoring=scoring
            )
            train_sizes = result["train_sizes"]
            metrics[scoring] = {k: v for k, v in result.items() if k != "train_sizes"}

        data = {
            "random_state": random_state,
            "train_sizes": train_sizes,
            "metrics": metrics,
        }

        ensure_dir(os.path.dirname(filepath))
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def _cv_train_evaluate(
        self,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: tuple,
        random_state: int,
        model_filepath: str,
        metrics_filepath: str,
        learning_curve_path: str,
        classifier_model: ClassifierModel,
        shap_explanation_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Steps 4-5: GridSearchCV training then evaluation on the held-out test set.

        Fits the configured classifier via GridSearchCV on the selected training
        features, saves the best model to disk, evaluates it on the held-out test
        set, and writes per-seed metrics JSON.

        Args:
            y_train (pd.Series): Resampled training labels.
            X_test (pd.DataFrame): Held-out test features (pre-selection).
            y_test (pd.Series): Held-out test labels.
            selected_features (tuple): Return value of :meth:`select_features`
                containing (df_selected, top_features, all_features, n_features).
            random_state (int): Random seed used for the classifier.
            model_filepath (str): Path to save the trained model pickle.
            metrics_filepath (str): Path to save the per-seed metrics JSON.
            learning_curve_path (str): Path to save the per-seed learning curve JSON.
            classifier_model (ClassifierModel): ClassifierModel instance to use.
            shap_explanation_filepath (str | None, optional): Path for the SHAP
                explanation cache. Defaults to None.

        Returns:
            dict[str, Any]: Metrics dictionary produced by ModelEvaluator.
        """
        features_train_resampled_selected, top_selected_features, _, _ = (
            selected_features
        )
        top_n_features = top_selected_features.index.tolist()
        X_test_selected = X_test[top_n_features]

        classifier, grid_search, best_model = self._setup_grid_search(
            random_state,
            features_train_resampled_selected,
            y_train,
            top_n_features,
            classifier_model=classifier_model,
        )

        joblib.dump(best_model, model_filepath)
        if self.verbose:
            logger.info(f"{random_state:05d}: Model at {model_filepath}")

        try:
            self._compute_all_learning_curves(
                estimator=best_model,
                X_train=features_train_resampled_selected[top_n_features],
                y_train=y_train,
                random_state=random_state,
                filepath=learning_curve_path,
            )
        except Exception as e:
            logger.warning(
                f"Seed {random_state:05d}: learning curve computation failed. Reason: {e}"
            )
            learning_curve_path = None  # type: ignore[assignment]

        # Evaluating model
        model_evaluator = ModelEvaluator(
            random_state=random_state,
            model=grid_search,
            X_test=X_test_selected,
            y_test=y_test,
            model_name=classifier_model.name,
            output_dir=self.classifier_dirs[classifier_model.slug_name],
            plot_shap=self.plot_shap,
            selected_features=top_n_features,
            shap_explanation_filepath=shap_explanation_filepath,
            learning_curve_path=learning_curve_path,
        )

        metrics: dict[str, Any] = model_evaluator.get_metrics()

        grid_params = grid_search.best_params_
        best_params = {f"best_params_{k}": v for k, v in grid_params.items()}
        metrics.update(
            {
                "cv_strategy": classifier.cv_strategy,
                "random_state": random_state,
                "best_cv_score": grid_search.best_score_,
                **best_params,
            }
        )

        with open(metrics_filepath, "w") as f:
            json.dump(metrics, f, indent=4)

        try:
            model_evaluator.plot_all(dpi=150)
        except Exception as e:
            logger.warning(
                f"Seed {random_state:05d}: plot_all() failed and plots were skipped. "
                f"Reason: {e}"
            )

        if self.verbose:
            logger.info(f"{random_state:05d}: Metrics at {metrics_filepath}")
            logger.info(
                f"Seed {random_state:05d} - Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            )

        return metrics

    def _run_train_and_evaluate(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> dict[str, tuple[int, str, str, dict, str, str, str]] | None:
        """Train feature selection and all classifiers for a single random seed (eval mode).

        Orchestrates the per-seed pipeline:

        1–2. Under-sampling and train/test split (ONCE per seed).
        3.   Feature selection on training set (ONCE per seed, shared across classifiers).
        4–5. For each classifier: GridSearchCV training and evaluation on held-out test set.

        Args:
            random_state (int): Base random state value.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            dict[str, tuple]: Keyed by classifier slug. Each value is a 7-tuple of
                (random_state, significant_filepath, model_filepath, metrics,
                X_test_filepath, y_test_filepath, shap_explanation_filepath).
            None: When features reduced to zero.
        """
        if self.debug:
            logger.debug(f"_run_train_and_evaluate: seed={random_state}")

        # ========== STEP 0: Shared filepath preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            X_test_filepath,
            y_test_filepath,
            _,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        logger.info(f"Training Seed: {random_state:05d}")

        # ========== STEPS 1-2: Train/Test Split + Resample (ONCE) ==========
        X_train, X_test, y_train, y_test = self._split_and_resample(
            X=self.df_features,
            y=self.df_labels,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )

        # ========== STEP 3: Feature Selection (ONCE, shared) ==========
        result = self._select_features_for_seed(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

        if result is None:
            return None

        # Save held-out test data ONCE (shared across all classifiers)
        _, top_selected_features, _, _ = result
        X_test[top_selected_features.index.tolist()].to_csv(X_test_filepath)
        y_test.to_csv(y_test_filepath)

        # ========== STEPS 4-5: Per-classifier GridSearchCV + Evaluation ==========
        seed_results: dict[str, tuple] = {}
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            (
                model_filepath,
                metrics_filepath,
                shap_explanation_filepath,
                learning_curve_path,
            ) = self._generate_classifier_filepaths(random_state, classifier_slug)

            metrics = self._cv_train_evaluate(
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=result,
                random_state=random_state,
                model_filepath=model_filepath,
                metrics_filepath=metrics_filepath,
                learning_curve_path=learning_curve_path,
                classifier_model=classifier_model,
                shap_explanation_filepath=shap_explanation_filepath,
            )

            seed_results[classifier_slug] = (
                random_state,
                significant_filepath,
                model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            )

        return seed_results

    def _collect_pending_evaluate_jobs(
        self,
        random_states: list[int],
        sampling_strategy: str | float,
        save_all_features: bool,
        plot_significant_features: bool,
    ) -> tuple[list[tuple], dict[str, list[dict]], dict[str, list[dict]]]:
        """Identify seeds that need training and load results for already-completed seeds.

        Iterates over all random states, checks whether every classifier's model,
        metrics, and significant-features files already exist on disk, and either
        loads the cached results or queues the seed for training.

        ``_generate_classifier_filepaths`` is called exactly once per classifier per
        seed; the returned paths are reused for both the existence check and the
        record/metrics building, avoiding redundant filesystem path construction.

        Args:
            random_states (list[int]): Ordered list of random seed values to inspect.
            sampling_strategy (str | float): Under-sampling ratio forwarded to pending jobs.
            save_all_features (bool): Whether to save all ranked features per seed.
            plot_significant_features (bool): Whether to generate feature importance plots.

        Returns:
            tuple: A 3-tuple of:

                - **pending_jobs** (list[tuple]): Seeds that still need training,
                  each as ``(random_state, sampling_strategy, save_all_features,
                  plot_significant_features)``.
                - **records_per_classifier** (dict[str, list[dict]]): Already-completed
                  seed records keyed by classifier slug.
                - **all_metrics** (dict[str, list[dict]]): Loaded metrics dicts for
                  already-completed seeds, keyed by classifier slug.
        """
        records_per_classifier: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }
        all_metrics: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }
        pending_jobs: list[tuple] = []

        for _rs in random_states:
            (
                _,
                _significant_filepath,
                _,
                _,
                _X_test_filepath,
                _y_test_filepath,
                _,
            ) = self._generate_shared_filepaths(_rs)

            # Compute classifier filepaths once per classifier and cache them
            _classifier_filepaths: dict[str, tuple[str, str, str]] = {
                classifier_model.slug_name: self._generate_classifier_filepaths(
                    _rs, classifier_model.slug_name
                )[:3]
                for classifier_model in self.classifier_models
            }

            _all_classifier_done = all(
                not self.overwrite
                and os.path.isfile(_significant_filepath)
                and os.path.isfile(_classifier_filepaths[classifier_model.slug_name][0])
                and os.path.isfile(_classifier_filepaths[classifier_model.slug_name][1])
                for classifier_model in self.classifier_models
            )

            if _all_classifier_done:
                logger.info(f"Seed {_rs:05d} already trained.")
                if _significant_filepath not in self.significant_features_csvs:
                    self.significant_features_csvs.append(_significant_filepath)
                for classifier_model in self.classifier_models:
                    classifier_slug = classifier_model.slug_name
                    _model_filepath, _metrics_filepath, _shap_filepath = (
                        _classifier_filepaths[classifier_slug]
                    )
                    with open(_metrics_filepath) as f:
                        metrics = json.load(f)
                    records_per_classifier[classifier_slug].append(
                        {
                            "random_state": _rs,
                            "significant_features_csv": _significant_filepath,
                            "trained_model_filepath": _model_filepath,
                            "X_test_filepath": _X_test_filepath,
                            "y_test_filepath": _y_test_filepath,
                            "shap_explanation_filepath": _shap_filepath,
                        }
                    )
                    all_metrics[classifier_slug].append(metrics)
            else:
                pending_jobs.append(
                    (
                        _rs,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )

        return pending_jobs, records_per_classifier, all_metrics

    def train_and_evaluate(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train feature selection and classifier models across multiple random seeds.

        For each seed, performs:

        1. Train/test split (80/20, stratified)
        2. Random under-sampling on training set only
        3. Feature selection on training set only (shared across all classifiers)
        4. For each classifier: training with GridSearchCV
        5. For each classifier: evaluation on held-out test set

        Under-sampling and feature selection are run ONCE per seed and reused
        across all classifiers, which significantly reduces redundant computation
        when training multiple classifiers simultaneously.

        Args:
            random_state (int, optional): Initial random state seed. Defaults to 0.
            total_seed (int, optional): Total number of seeds to run. Defaults to 500.
            sampling_strategy (str | float, optional): Under-sampling ratio for
                balancing classes. Defaults to 0.75.
            save_all_features (bool, optional): Save all features per seed,
                not just top-N. Defaults to False.
            plot_significant_features (bool, optional): Generate feature importance
                plots. Defaults to False.

        Example:
            >>> ModelTrainer(
            ...     extracted_features_csv="...",
            ...     label_features_csv="...",
            ...     classifiers=["rf", "xgb"],
            ... ).train_and_evaluate(random_state=42, total_seed=100)
        """
        self.create_directories()
        for slug in self.classifier_dirs:
            ensure_dir(self.metrics_dirs[slug])

        if save_all_features:
            ensure_dir(self.shared_all_features_dir)

        if plot_significant_features:
            ensure_dir(self.shared_figures_dir)

        # Pre-filter: collect already-completed seeds without re-running them
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        pending_jobs, records_per_classifier, all_metrics = (
            self._collect_pending_evaluate_jobs(
                random_states,
                sampling_strategy,
                save_all_features,
                plot_significant_features,
            )
        )

        for seed_results in self._run_jobs(self._run_train_and_evaluate, pending_jobs):
            if seed_results is None:  # Feature selection returned nothing
                continue
            # seed_results is dict[classifier_slug -> 7-tuple]
            for classifier_slug, (
                _random_state,
                significant_features_csv,
                trained_model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            ) in seed_results.items():
                if classifier_slug not in records_per_classifier:
                    records_per_classifier[classifier_slug] = []
                    all_metrics[classifier_slug] = []
                # Only append significant_features_csv once (same for all classifiers)
                if significant_features_csv not in self.significant_features_csvs:
                    self.significant_features_csvs.append(significant_features_csv)
                records_per_classifier[classifier_slug].append(
                    {
                        "random_state": _random_state,
                        "significant_features_csv": significant_features_csv,
                        "trained_model_filepath": trained_model_filepath,
                        "X_test_filepath": X_test_filepath,
                        "y_test_filepath": y_test_filepath,
                        "shap_explanation_filepath": shap_explanation_filepath,
                    }
                )
                all_metrics[classifier_slug].append(metrics)

        # Aggregate feature selection results (shared across classifiers)
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Save registry and metrics per classifier
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            if not records_per_classifier[classifier_slug]:
                continue
            suffix_filename = self._save_models_registry(
                records_per_classifier[classifier_slug],
                random_state,
                total_seed,
                classifier_slug=classifier_slug,
            )
            self._aggregate_metrics(
                all_metrics[classifier_slug],
                suffix_filename=suffix_filename,
                classifier_slug=classifier_slug,
            )

        if self.verbose:
            logger.info(f"Models saved to: {self.csv}")

        return None

    def _aggregate_metrics(
        self,
        all_metrics: list[dict],
        suffix_filename: str = "",
        classifier_slug: str = "",
    ) -> None:
        """Aggregate metrics across all seeds for one classifier.

        Computes mean and std for each metric and saves to CSV.
        Also saves all individual metrics for detailed analysis.

        Args:
            all_metrics (list[dict]): List of metric dictionaries, one per seed.
            suffix_filename (str, optional): Suffix appended to output filenames.
                Defaults to ``""``.
            classifier_slug (str): Slugified classifier name used to resolve the correct
                output directory from ``self.classifier_dirs`` and ``self.metrics_dirs``.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.train_and_evaluate(total_seed=100)
            >>> # Creates: all_metrics_{suffix}.csv and metrics_summary_{suffix}.csv
        """
        classifier_dir = self.classifier_dirs[classifier_slug]
        metrics_dir = self.metrics_dirs[classifier_slug]
        ensure_dir(metrics_dir)

        df_metrics = pd.DataFrame(all_metrics)

        # Calculate summary statistics
        summary = df_metrics.describe().T
        summary_filepath = os.path.join(
            classifier_dir, f"metrics_summary_{suffix_filename}.csv"
        )
        summary.to_csv(summary_filepath)

        # Save all individual metrics
        all_metrics_filepath = os.path.join(
            classifier_dir, f"all_metrics_{suffix_filename}.csv"
        )
        df_metrics = df_metrics.set_index("random_state")
        df_metrics.to_csv(all_metrics_filepath, index=True)

        logger.info("=" * 60)
        logger.info("Metrics Summary (mean ± std across seeds)")
        logger.info("=" * 60)
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
        ]:
            mean = df_metrics[metric].mean()
            std = df_metrics[metric].std()
            logger.info(f"{metric:20s}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 60)

        if self.verbose:
            logger.info(f"Summary metrics saved to: {summary_filepath}")
            logger.info(f"All metrics saved to: {all_metrics_filepath}")
