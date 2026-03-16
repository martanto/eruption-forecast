"""Evaluation-mode trainer that extends BaseModelTrainer with per-seed metrics.

Adds the full evaluation pipeline on top of :class:`BaseModelTrainer`:
train/test split, GridSearchCV training, held-out test evaluation, learning
curves, SHAP, and metric aggregation across seeds.
"""

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
    """Extends BaseModelTrainer with evaluate() and its helper methods.

    Adds the full evaluation pipeline on top of :class:`BaseModelTrainer`:
    per-seed stratified train/test split, random under-sampling on the training
    set, feature selection, GridSearchCV training, held-out test evaluation,
    learning-curve computation, SHAP explanation caching, and metric aggregation
    across all seeds.

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

    def _run_shared_evaluate(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> str | None:
        """Phase 1 - Feature Selection worker: split, resample, select features, and save test data to disk.

        Performs the shared per-seed work for evaluate mode:
        train/test split, random under-sampling on the training set, and
        feature selection. Saves the top-N significant features CSV and the
        held-out test split to disk so Phase 2 workers can load them without
        repeating the expensive selection step.

        This method is designed to be called in parallel across seeds via
        _run_jobs(). It is deterministic for a given random_state.

        Args:
            random_state (int): Random seed controlling split, resampling, and
                feature selection.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save full ranked feature list.
                Defaults to False.
            plot_features (bool, optional): Generate feature importance plots.
                Defaults to False.

        Returns:
            str: Path to the significant features CSV if feature selection
                produced at least one feature.
            None: If features collapsed to zero after selection.
        """
        if self.debug:
            logger.debug(f"_run_shared_evaluate: seed={random_state}")

        (
            _,
            significant_filepath,
            tsfresh_filepath,
            all_features_filepath,
            all_figures_filepath,
            X_test_filepath,
            y_test_filepath,
            can_skip_shared,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        # Short-circuit if shared work was already done in a previous run,
        # but only if all required shared artifacts actually exist.
        # For "combined" method, Phase 1 completion is marked by tsfresh_filepath.
        phase1_marker = tsfresh_filepath
        can_skip = (
            can_skip_shared
            and os.path.exists(phase1_marker)
            and os.path.exists(X_test_filepath)
            and os.path.exists(y_test_filepath)
            and (
                not save_features
                or (
                    all_features_filepath is not None
                    and os.path.exists(all_features_filepath)
                )
            )
            and (
                not plot_features
                or (
                    all_figures_filepath is not None
                    and os.path.exists(all_figures_filepath)
                )
            )
        )
        if can_skip:
            logger.info(f"Seed {random_state:05d}: shared work already done, skipping.")
            return tsfresh_filepath

        logger.info(f"Training Seed: {random_state:05d}")

        X_train, X_test, y_train, y_test = self._split_and_resample(
            X=self.df_features,
            y=self.df_labels,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )

        # For "combined" method, select_features() writes tsfresh-filtered features to
        # tsfresh_filepath; for other methods it writes top-N features to significant_filepath.
        phase1_output_filepath = tsfresh_filepath
        result = self.select_features(
            features=X_train,
            labels=y_train,
            random_state=random_state,
            significant_filepath=phase1_output_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

        if result is None:
            return None

        # Save held-out test data once — shared across all classifiers for this seed.
        # Use all available feature names from Phase 1 output so X_test is not narrowed
        # before Phase 2 applies per-classifier RF importance selection.
        df_selected, _, _, _ = result
        phase1_feature_names = df_selected.columns.tolist()
        X_test[phase1_feature_names].to_csv(X_test_filepath)
        y_test.to_csv(y_test_filepath)

        return tsfresh_filepath

    def _run_classify_and_evaluate(
        self,
        random_state: int,
        classifier_slug: str,
        sampling_strategy: str | float = 0.75,
    ) -> tuple | None:
        """Phase 2 worker: train and evaluate one (seed, classifier) pair.

        Reconstructs the training data deterministically by re-running the
        split and resample with the same random_state used in Phase 1 - Feature Selection. Loads
        the pre-selected top-N features from the significant features CSV
        written by _run_shared_evaluate(), then trains the specified classifier
        via GridSearchCV and evaluates it on the held-out test set.

        This method is designed to be called in parallel across all
        (seed, classifier) combinations via _run_jobs().

        Args:
            random_state (int): Random seed matching the Phase 1 - Feature Selection run for this seed.
            classifier_slug (str): Slug name of the classifier to train.
            sampling_strategy (str | float, optional): Under-sampling ratio,
                must match the value used in Phase 1 - Feature Selection. Defaults to 0.75.

        Returns:
            tuple: An 8-tuple of (classifier_slug, random_state,
                significant_filepath, model_filepath, metrics, X_test_filepath,
                y_test_filepath, shap_explanation_filepath).
            None: If the significant features file is missing or empty.
        """
        if self.debug:
            logger.debug(
                f"_run_classify_and_evaluate: seed={random_state}, clf={classifier_slug}"
            )

        (
            _,
            significant_filepath,
            tsfresh_filepath,
            _,
            _,
            X_test_filepath,
            y_test_filepath,
            _,
        ) = self._generate_shared_filepaths(random_state=random_state)

        # Phase 1 completion marker: tsfresh_filepath for "combined", else significant_filepath.
        phase1_filepath = tsfresh_filepath
        if not os.path.isfile(phase1_filepath):
            logger.warning(
                f"Seed {random_state:05d}: Phase 1 features file missing, skipping."
            )
            return None

        (
            model_filepath,
            metrics_filepath,
            shap_explanation_filepath,
            learning_curve_path,
            classifier_features_filepath,
        ) = self._generate_classifier_filepaths(random_state, classifier_slug)

        # For "combined" method, the final per-classifier top-N features CSV is
        # classifier_features_filepath. For other methods use significant_filepath.
        if self.feature_selection_method == "combined":
            registry_features_filepath = classifier_features_filepath
        else:
            registry_features_filepath = significant_filepath

        # Return cached result without re-training if outputs already exist.
        if (
            not self.overwrite
            and os.path.isfile(model_filepath)
            and os.path.isfile(metrics_filepath)
        ):
            logger.info(
                f"Seed {random_state:05d} / {classifier_slug}: already trained, skipping."
            )
            with open(metrics_filepath) as f:
                metrics = json.load(f)
            return (
                classifier_slug,
                random_state,
                registry_features_filepath,
                model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            )

        # Deterministically reproduce the same split and resample as Phase 1.
        X_train, X_test, y_train, y_test = self._split_and_resample(
            X=self.df_features,
            y=self.df_labels,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )

        if self.feature_selection_method == "combined":
            # Load tsfresh-filtered features from Phase 1.
            tsfresh_feature_names = pd.read_csv(
                tsfresh_filepath, index_col=0
            ).index.tolist()
            if not tsfresh_feature_names:
                return None

            classifier_model = next(
                m for m in self.classifier_models if m.slug_name == classifier_slug
            )

            # Run GridSearchCV on tsfresh features first; afterwards reuse the best
            # estimator for RF importance selection to skip a redundant RF training.
            tsfresh_selected = (
                X_train[tsfresh_feature_names],
                pd.Series(index=tsfresh_feature_names, dtype=float),
                None,
                len(tsfresh_feature_names),
            )
            metrics = self._cv_train_evaluate(
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=tsfresh_selected,
                random_state=random_state,
                model_filepath=model_filepath,
                metrics_filepath=metrics_filepath,
                learning_curve_path=learning_curve_path,
                classifier_model=classifier_model,
                shap_explanation_filepath=shap_explanation_filepath,
            )

            # Load the just-saved best model to use as the RF importance estimator.
            best_model = joblib.load(model_filepath)
            # Use the best model only when it is a tree-based estimator compatible
            # with permutation_importance (all sklearn classifiers qualify).
            top_feature_names, _ = self._apply_rf_importance_selection(
                X=X_train[tsfresh_feature_names],
                y=y_train,
                top_n=self.number_of_significant_features,
                random_state=random_state,
                classifier_features_filepath=classifier_features_filepath,
                estimator=best_model,
            )

            return (
                classifier_slug,
                random_state,
                classifier_features_filepath,
                model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            )

        # Non-combined path: load pre-selected top-N features from Phase 1.
        top_n_features = pd.read_csv(significant_filepath, index_col=0).index.tolist()
        if not top_n_features:
            return None

        classifier_model = next(
            m for m in self.classifier_models if m.slug_name == classifier_slug
        )

        # Reconstruct the selected_features tuple expected by _cv_train_evaluate.
        selected_features = (
            X_train[top_n_features],
            pd.Series(index=top_n_features, dtype=float),
            None,
            len(top_n_features),
        )

        metrics = self._cv_train_evaluate(
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selected_features=selected_features,
            random_state=random_state,
            model_filepath=model_filepath,
            metrics_filepath=metrics_filepath,
            learning_curve_path=learning_curve_path,
            classifier_model=classifier_model,
            shap_explanation_filepath=shap_explanation_filepath,
        )

        return (
            classifier_slug,
            random_state,
            significant_filepath,
            model_filepath,
            metrics,
            X_test_filepath,
            y_test_filepath,
            shap_explanation_filepath,
        )

    def _collect_pending_evaluate_jobs(
        self,
        random_states: list[int],
        sampling_strategy: str | float,
        save_all_features: bool,
        plot_significant_features: bool,
    ) -> tuple[list[tuple], list[tuple], dict[str, list[dict]], dict[str, list[dict]]]:
        """Identify seeds and (seed, classifier) pairs that still need work.

        Iterates over all random states and, for each seed, checks whether the
        shared significant-features file exists. If it does, each classifier
        is checked individually so partially-completed seeds are handled
        correctly — only the missing (seed, classifier) pairs are queued for
        Phase 2.

        Args:
            random_states (list[int]): Ordered list of random seed values to inspect.
            sampling_strategy (str | float): Under-sampling ratio forwarded to
                pending Phase 1 - Feature Selection jobs.
            save_all_features (bool): Whether to save all ranked features per seed.
            plot_significant_features (bool): Whether to generate feature importance
                plots.

        Returns:
            tuple: A 4-tuple of:

                - **pending_feature_selection_jobs** (list[tuple]): Seeds missing shared work,
                  each as ``(random_state, sampling_strategy, save_all_features,
                  plot_significant_features)``.
                - **pending_evaluate_model_jobs** (list[tuple]): (seed, classifier) pairs
                  missing classifier outputs, each as
                  ``(random_state, classifier_slug, sampling_strategy)``.
                - **records_per_classifier** (dict[str, list[dict]]): Already-completed
                  seed records keyed by classifier slug.
                - **all_metrics** (dict[str, list[dict]]): Loaded metrics dicts for
                  already-completed seeds, keyed by classifier slug.
        """
        records_per_classifier: dict[str, list[dict]] = {
            m.slug_name: [] for m in self.classifier_models
        }
        all_metrics: dict[str, list[dict]] = {
            m.slug_name: [] for m in self.classifier_models
        }
        pending_feature_selection_jobs: list[tuple] = []
        pending_evaluate_model_jobs: list[tuple] = []

        for random_state in random_states:
            (
                _,
                _significant_filepath,
                _tsfresh_filepath,
                _,
                _,
                _X_test_filepath,
                _y_test_filepath,
                _,
            ) = self._generate_shared_filepaths(random_state)

            # Phase 1 completion is marked by tsfresh_filepath (for "combined" method)
            # or significant_filepath (for other methods).  Both are the same file
            # for non-combined methods, so this check is always correct.
            _phase1_filepath = _tsfresh_filepath
            if self.overwrite or not (
                os.path.isfile(_phase1_filepath)
                and os.path.isfile(_X_test_filepath)
                and os.path.isfile(_y_test_filepath)
            ):
                pending_feature_selection_jobs.append(
                    (
                        random_state,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )
                continue

            # Shared work done — check each classifier independently.
            for classifier_model in self.classifier_models:
                classifier_slug = classifier_model.slug_name
                _model_filepath, _metrics_filepath, _shap_filepath, _, _clf_features_filepath = (
                    self._generate_classifier_filepaths(random_state, classifier_slug)
                )

                # For "combined" method the registry stores classifier_features_filepath;
                # for other methods it stores significant_filepath.
                if self.feature_selection_method == "combined":
                    _registry_features_filepath = _clf_features_filepath
                else:
                    _registry_features_filepath = _significant_filepath

                if (
                    not self.overwrite
                    and os.path.isfile(_model_filepath)
                    and os.path.isfile(_metrics_filepath)
                ):
                    with open(_metrics_filepath) as f:
                        metrics = json.load(f)
                    records_per_classifier[classifier_slug].append(
                        {
                            "random_state": random_state,
                            "significant_features_csv": _registry_features_filepath,
                            "trained_model_filepath": _model_filepath,
                            "X_test_filepath": _X_test_filepath,
                            "y_test_filepath": _y_test_filepath,
                            "shap_explanation_filepath": _shap_filepath,
                        }
                    )
                    all_metrics[classifier_slug].append(metrics)
                else:
                    pending_evaluate_model_jobs.append(
                        (random_state, classifier_slug, sampling_strategy)
                    )

        return (
            pending_feature_selection_jobs,
            pending_evaluate_model_jobs,
            records_per_classifier,
            all_metrics,
        )

    def evaluate(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train feature selection and classifier models across multiple random seeds.

        Uses a two-phase parallel dispatch to maximise CPU utilisation:

        - **Phase 1 - Feature Selection** (parallel across seeds): train/test split, random
          under-sampling, and feature selection. Results are saved to disk.
        - **Phase 2 - Evaluate Model** (parallel across seed × classifier pairs): GridSearchCV
          training and held-out test evaluation for every (seed, classifier)
          combination. Training data is reconstructed deterministically via
          the fixed ``random_state``.

        Under-sampling and feature selection are run ONCE per seed (Phase 1 - Feature Selection)
        and reused across all classifiers (Phase 2), which significantly
        reduces redundant computation when training multiple classifiers.

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
            ... ).evaluate(random_state=42, total_seed=100)
        """
        self.with_evaluation = True
        # Re-derive all output paths now that with_evaluation is True so that
        # artefacts land under "evaluations/" instead of "predictions/".
        (
            self.output_dir,
            self.shared_features_dir,
            self.shared_significant_dir,
            self.shared_all_features_dir,
            self.shared_figures_dir,
            self.shared_tests_dir,
            self.classifier_dirs,
            self.models_dirs,
            self.metrics_dirs,
        ) = self.set_directories(os.path.dirname(self.output_dir))
        self.create_directories(
            save_all_features, plot_significant_features, with_evaluation=True
        )

        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        (
            pending_feature_selection_jobs,
            pending_evaluate_model_jobs,
            records_per_classifier,
            all_metrics,
        ) = self._collect_pending_evaluate_jobs(
            random_states,
            sampling_strategy,
            save_all_features,
            plot_significant_features,
        )

        # ===== PHASE 1: Parallel feature selection across seeds =====
        feature_selection_results: list[str | None] = self._run_jobs(
            self._run_shared_evaluate, pending_feature_selection_jobs
        )

        # Build Phase 2 jobs for seeds that completed Phase 1 successfully.
        new_evaluate_model_jobs: list[tuple] = []
        for result_path, job in zip(
            feature_selection_results, pending_feature_selection_jobs, strict=True
        ):
            if result_path is None:
                continue  # Feature selection produced 0 features — skip all classifiers.
            _rs = job[0]
            for classifier_model in self.classifier_models:
                new_evaluate_model_jobs.append(
                    (_rs, classifier_model.slug_name, sampling_strategy)
                )

        # ===== PHASE 2: Parallel (seed × classifier) training + evaluation =====
        all_evaluate_model_jobs = pending_evaluate_model_jobs + new_evaluate_model_jobs
        evaluate_model_results = self._run_jobs(
            self._run_classify_and_evaluate, all_evaluate_model_jobs
        )

        for result in evaluate_model_results:
            if result is None:
                continue
            (
                classifier_slug,
                _random_state,
                significant_features_csv,
                trained_model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            ) = result
            if classifier_slug not in records_per_classifier:
                records_per_classifier[classifier_slug] = []
                all_metrics[classifier_slug] = []
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

        # Collect significant_features_csvs for all completed seeds.
        # For "combined" method the tsfresh_filepath (index 2) holds the Phase 1
        # feature set written once per seed; for other methods index 2 == index 1.
        for _rs in random_states:
            sf = self._generate_shared_filepaths(_rs)[2]
            if os.path.isfile(sf) and sf not in self.significant_features_csvs:
                self.significant_features_csvs.append(sf)

        # Aggregate feature selection results (shared across classifiers).
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Save registry and metrics per classifier.
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
            >>> trainer.evaluate(total_seed=100)
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
