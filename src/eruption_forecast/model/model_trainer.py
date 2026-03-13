import os
from typing import Self

import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import random_under_sampler
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.evaluation_trainer import EvaluationTrainer


class ModelTrainer(EvaluationTrainer):
    """Train feature-selection and classifier models over multiple random seeds.

    Loads pre-extracted features and labels, then for each random seed performs:

    1. Train/test split (80/20, stratified) to prevent data leakage
    2. Random under-sampling on training set only to balance classes
    3. Feature selection on training set using tsfresh relevance filtering (ONCE per seed)
    4. For each classifier: GridSearchCV training and cross-validation
    5. Evaluation on held-out test set (when using train_and_evaluate)

    Both ``train_and_evaluate`` and ``train`` use a two-phase parallel dispatch:

    - **Phase 1** (parallel across seeds): shared per-seed work (split/resample +
      feature selection). Results are saved to disk.
    - **Phase 2** (parallel across seed × classifier pairs): one GridSearchCV job
      per (seed, classifier) combination. Training data is reconstructed
      deterministically via the fixed ``random_state``.

    Use :meth:`train_and_evaluate` for 80/20 split with held-out evaluation metrics,
    or :meth:`train` for full-dataset training (no metrics) intended for production.
    Call :meth:`fit` to dispatch between the two modes via the ``with_evaluation`` flag.

    All constructor arguments are forwarded to :class:`EvaluationTrainer` and
    :class:`BaseModelTrainer`. See :class:`BaseModelTrainer` for the full parameter
    and attribute documentation.

    Examples:
        >>> # Train with shared feature selection across rf and xgb
        >>> trainer = ModelTrainer(
        ...     extracted_features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     classifiers=["rf", "xgb"],
        ...     output_dir="output/trainings",
        ...     n_jobs=4,
        ... )
        >>> trainer.fit(with_evaluation=True, random_state=0, total_seed=100)
    """

    def _run_shared_train(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> str | None:
        """Phase 1 worker: resample the full dataset and select features for train mode.

        Performs the shared per-seed work for train (no evaluation) mode:
        random under-sampling on the full dataset and feature selection. Saves
        the top-N significant features CSV to disk so Phase 2 workers can load
        them without repeating the expensive selection step.

        This method is designed to be called in parallel across seeds via
        _run_jobs(). It is deterministic for a given random_state.

        Args:
            random_state (int): Random seed controlling resampling and
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
            logger.debug(f"_run_shared_train: seed={random_state}")

        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            _,
            _,
            can_skip_shared,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        # Short-circuit if shared work was already done in a previous run.
        if can_skip_shared:
            logger.info(f"Seed {random_state:05d}: shared work already done, skipping.")
            return significant_filepath

        features_resampled, labels_resampled = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        result = self.select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

        if result is None:
            return None

        return significant_filepath

    def _run_train_classifier(
        self,
        random_state: int,
        classifier_slug: str,
        sampling_strategy: str | float = 0.75,
    ) -> tuple | None:
        """Phase 2 worker: train one (seed, classifier) pair in train-only mode.

        Reconstructs the resampled training data deterministically by re-running
        random under-sampling with the same random_state used in Phase 1. Loads
        the pre-selected top-N features from the significant features CSV written
        by _run_shared_train(), then trains the specified classifier via
        GridSearchCV and saves the model.

        No evaluation metrics are computed. This method is designed to be called
        in parallel across all (seed, classifier) combinations via _run_jobs().

        Args:
            random_state (int): Random seed matching the Phase 1 run for this seed.
            classifier_slug (str): Slug name of the classifier to train.
            sampling_strategy (str | float, optional): Under-sampling ratio,
                must match the value used in Phase 1. Defaults to 0.75.

        Returns:
            tuple: A 4-tuple of (classifier_slug, random_state,
                significant_filepath, model_filepath).
            None: If the significant features file is missing or empty.
        """
        if self.debug:
            logger.debug(
                f"_run_train_classifier: seed={random_state}, clf={classifier_slug}"
            )

        (
            _,
            significant_filepath,
            _,
            _,
            _,
            _,
            _,
        ) = self._generate_shared_filepaths(random_state=random_state)

        if not os.path.isfile(significant_filepath):
            logger.warning(
                f"Seed {random_state:05d}: significant features file missing, skipping."
            )
            return None

        top_n_features = pd.read_csv(significant_filepath, index_col=0).index.tolist()
        if not top_n_features:
            return None

        classifier_model = next(
            m for m in self.classifier_models if m.slug_name == classifier_slug
        )

        model_filepath = self._get_model_filepath(random_state, classifier_slug)

        # Return cached result without re-training if the model already exists.
        if not self.overwrite and os.path.isfile(model_filepath):
            logger.info(
                f"Seed {random_state:05d} / {classifier_slug}: model exists, skipping."
            )
            return classifier_slug, random_state, significant_filepath, model_filepath

        # Deterministically reproduce the same resample as Phase 1.
        features_resampled, labels_resampled = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        logger.info(f"Fitting Seed: {random_state:05d} / {classifier_slug}")

        _, _, best_model = self._setup_grid_search(
            random_state,
            features_resampled,
            labels_resampled,
            top_n_features,
            classifier_model=classifier_model,
        )

        joblib.dump(best_model, model_filepath)

        if self.verbose:
            logger.info(f"Model {random_state:05d}: {model_filepath}")

        return classifier_slug, random_state, significant_filepath, model_filepath

    def _collect_pending_train_jobs(
        self,
        random_states: list[int],
        sampling_strategy: str | float,
        save_all_features: bool,
        plot_significant_features: bool,
    ) -> tuple[list[tuple], list[tuple], dict[str, list[dict]]]:
        """Identify seeds and (seed, classifier) pairs that still need work for train mode.

        Iterates over all random states and checks whether the shared
        significant-features file exists for each seed. If it does, each
        classifier is checked independently so that partially-completed seeds
        are handled correctly — only the missing (seed, classifier) pairs are
        queued for Phase 2.

        Args:
            random_states (list[int]): Ordered list of random seed values to inspect.
            sampling_strategy (str | float): Under-sampling ratio forwarded to
                pending Phase 1 jobs.
            save_all_features (bool): Whether to save all ranked features per seed.
            plot_significant_features (bool): Whether to generate feature importance
                plots.

        Returns:
            tuple: A 3-tuple of:

                - **pending_phase1_jobs** (list[tuple]): Seeds missing shared work,
                  each as ``(random_state, sampling_strategy, save_all_features,
                  plot_significant_features)``.
                - **pending_phase2_jobs** (list[tuple]): (seed, classifier) pairs
                  missing model files, each as
                  ``(random_state, classifier_slug, sampling_strategy)``.
                - **records_per_classifier** (dict[str, list[dict]]): Already-completed
                  seed records keyed by classifier slug.
        """
        records_per_classifier: dict[str, list[dict]] = {
            m.slug_name: [] for m in self.classifier_models
        }
        pending_phase1_jobs: list[tuple] = []
        pending_phase2_jobs: list[tuple] = []

        for _rs in random_states:
            (
                _,
                _significant_filepath,
                _,
                _,
                _,
                _,
                _,
            ) = self._generate_shared_filepaths(_rs)

            # Shared work not done — queue Phase 1; no Phase 2 jobs possible yet.
            if self.overwrite or not os.path.isfile(_significant_filepath):
                pending_phase1_jobs.append(
                    (
                        _rs,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )
                continue

            # Shared work done — check each classifier independently.
            for classifier_model in self.classifier_models:
                classifier_slug = classifier_model.slug_name
                _model_filepath = self._get_model_filepath(_rs, classifier_slug)

                if not self.overwrite and os.path.isfile(_model_filepath):
                    records_per_classifier[classifier_slug].append(
                        {
                            "random_state": _rs,
                            "significant_features_csv": _significant_filepath,
                            "trained_model_filepath": _model_filepath,
                        }
                    )
                else:
                    pending_phase2_jobs.append(
                        (_rs, classifier_slug, sampling_strategy)
                    )

        return pending_phase1_jobs, pending_phase2_jobs, records_per_classifier

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train on the full dataset across multiple seeds (no train/test split).

        Uses a two-phase parallel dispatch:

        - **Phase 1** (parallel across seeds): random under-sampling and feature
          selection on the full dataset. Results are saved to disk.
        - **Phase 2** (parallel across seed × classifier pairs): one GridSearchCV
          job per (seed, classifier) combination. Training data is reconstructed
          deterministically via the fixed ``random_state``.

        Unlike ``train_and_evaluate()``, this method does NOT perform a train/test
        split and does NOT compute evaluation metrics. It is intended for final model
        training when a separate "future" dataset will be used for evaluation
        via ``ModelPredictor``.

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
            >>> trainer = ModelTrainer(
            ...     extracted_features_csv="output/features/extracted_features.csv",
            ...     label_features_csv="output/features/label_features.csv",
            ...     classifiers=["rf", "xgb"],
            ... )
            >>> trainer.train(random_state=0, total_seed=5)
        """
        # Since we are not using evaluation, change the folder name from
        # ``evaluations`` to ``predictions``
        output_dir = self.output_dir.replace("evaluations", "predictions")

        # Update current directories with new output directory
        self.update_directories(output_dir=output_dir)

        if save_all_features:
            ensure_dir(self.shared_all_features_dir)

        if plot_significant_features:
            ensure_dir(self.shared_figures_dir)

        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        pending_phase1_jobs, pending_phase2_jobs, records_per_classifier = (
            self._collect_pending_train_jobs(
                random_states,
                sampling_strategy,
                save_all_features,
                plot_significant_features,
            )
        )

        # ===== PHASE 1: Parallel feature selection across seeds =====
        phase1_results: list[str | None] = self._run_jobs(
            self._run_shared_train, pending_phase1_jobs
        )

        # Build Phase 2 jobs for seeds that completed Phase 1 successfully.
        new_phase2_jobs: list[tuple] = []
        for result_path, job in zip(phase1_results, pending_phase1_jobs, strict=True):
            if result_path is None:
                continue  # Feature selection produced 0 features — skip all classifiers.
            _rs = job[0]
            for classifier_model in self.classifier_models:
                new_phase2_jobs.append(
                    (_rs, classifier_model.slug_name, sampling_strategy)
                )

        # ===== PHASE 2: Parallel (seed × classifier) training =====
        all_phase2_jobs = pending_phase2_jobs + new_phase2_jobs
        phase2_results = self._run_jobs(self._run_train_classifier, all_phase2_jobs)

        for result in phase2_results:
            if result is None:
                continue
            (
                classifier_slug,
                _random_state,
                significant_features_csv,
                trained_model_filepath,
            ) = result
            if classifier_slug not in records_per_classifier:
                records_per_classifier[classifier_slug] = []
            if significant_features_csv not in self.significant_features_csvs:
                self.significant_features_csvs.append(significant_features_csv)
            records_per_classifier[classifier_slug].append(
                {
                    "random_state": _random_state,
                    "significant_features_csv": significant_features_csv,
                    "trained_model_filepath": trained_model_filepath,
                }
            )

        # Collect significant_features_csvs for all completed seeds.
        for _rs in random_states:
            sf = self._generate_shared_filepaths(_rs)[1]
            if os.path.isfile(sf) and sf not in self.significant_features_csvs:
                self.significant_features_csvs.append(sf)

        # Aggregate feature selection results (shared)
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Save registry per classifier
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            if not records_per_classifier[classifier_slug]:
                continue
            self._save_models_registry(
                records_per_classifier[classifier_slug],
                random_state,
                total_seed,
                classifier_slug=classifier_slug,
            )

        if self.verbose:
            logger.info(f"Models saved to: {self.csv}")

        return None

    def fit(self, with_evaluation: bool = True, **kwargs) -> Self:
        """Dispatch to ``train_and_evaluate()`` or ``train()`` based on ``with_evaluation``.

        Args:
            with_evaluation (bool, optional): If True, calls ``train_and_evaluate()``
                (80/20 split + metrics). If False, calls ``train()`` (full dataset,
                no metrics). Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the chosen method.

        Returns:
            Self: The ModelTrainer instance for method chaining.
        """
        if with_evaluation:
            self.train_and_evaluate(**kwargs)
        else:
            self.train(**kwargs)
        return self
