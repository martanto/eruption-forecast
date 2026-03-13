import os
from typing import Self

import joblib

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

    Under-sampling and feature selection are shared across all classifiers for
    each seed, eliminating redundant computation when training multiple classifiers.

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

    def _run_train(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> dict[str, tuple[int, str, str]] | None:
        """Train on the full dataset (no train/test split) for a single random seed.

        Performs:

        1. Random under-sampling on full dataset (ONCE per seed).
        2. Feature selection on resampled data (ONCE per seed, shared).
        3. For each classifier: GridSearchCV training.
        4. For each classifier: save trained model.

        Args:
            random_state (int): Random seed for reproducibility.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            dict[str, tuple[int, str, str]]: Keyed by classifier slug. Each value is
                a 3-tuple of (random_state, significant_filepath, model_filepath).
            None: When features reduced to zero.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> seed_results = trainer._run_train(random_state=42)
        """
        if self.debug:
            logger.debug(f"_run_train: seed={random_state}")

        # ========== STEP 0: Shared filepath preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            _,
            _,
            _,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        # ========== STEP 1: Resample Full Dataset (ONCE) ==========
        features_resampled, labels_resampled = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        # ========== STEP 2: Feature Selection (ONCE, shared) ==========
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

        (
            features_resampled_selected,
            top_n_features_series,
            _all_selected_features,
            _n_features,
        ) = result

        top_n_features = top_n_features_series.index.tolist()

        # ========== STEP 3-4: Per-classifier GridSearchCV + Save ==========
        logger.info(f"Fitting Seed: {random_state:05d}")

        seed_results: dict[str, tuple] = {}
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            (
                model_filepath,
                _,
                _,
                _,
            ) = self._generate_classifier_filepaths(random_state, classifier_slug)

            _, _, best_model = self._setup_grid_search(
                random_state,
                features_resampled_selected,
                labels_resampled,
                top_n_features,
                classifier_model=classifier_model,
            )

            joblib.dump(best_model, model_filepath)

            if self.verbose:
                logger.info(f"Model {random_state:05d}: {model_filepath}")

            seed_results[classifier_slug] = (
                random_state,
                significant_filepath,
                model_filepath,
            )

        return seed_results

    def _collect_pending_train_jobs(
        self,
        random_states: list[int],
        sampling_strategy: str | float,
        save_all_features: bool,
        plot_significant_features: bool,
    ) -> tuple[list[tuple], dict[str, list[dict]]]:
        """Identify seeds that need training and load records for already-completed seeds.

        Iterates over all random states, checks whether every classifier's model and
        significant-features files already exist on disk, and either loads the cached
        records or queues the seed for training.

        :meth:`_get_model_filepath` is called exactly once per classifier per seed;
        the result is reused for both the existence check and the record building,
        avoiding redundant filesystem path construction.

        Args:
            random_states (list[int]): Ordered list of random seed values to inspect.
            sampling_strategy (str | float): Under-sampling ratio forwarded to pending jobs.
            save_all_features (bool): Whether to save all ranked features per seed.
            plot_significant_features (bool): Whether to generate feature importance plots.

        Returns:
            tuple: A 2-tuple of:

                - **pending_jobs** (list[tuple]): Seeds that still need training,
                  each as ``(random_state, sampling_strategy, save_all_features,
                  plot_significant_features)``.
                - **records_per_classifier** (dict[str, list[dict]]): Already-completed
                  seed records keyed by classifier slug.
        """
        records_per_classifier: dict[str, list[dict]] = {
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
                _,
                _,
                _,
            ) = self._generate_shared_filepaths(_rs)

            # Compute model filepath once per classifier and cache it
            _model_filepaths: dict[str, str] = {
                classifier_model.slug_name: self._get_model_filepath(
                    _rs, classifier_model.slug_name
                )
                for classifier_model in self.classifier_models
            }

            _all_classifier_done = all(
                not self.overwrite
                and os.path.isfile(_significant_filepath)
                and os.path.isfile(_model_filepaths[classifier_model.slug_name])
                for classifier_model in self.classifier_models
            )

            if _all_classifier_done:
                logger.info(f"Seed {_rs:05d} already fitted.")
                if _significant_filepath not in self.significant_features_csvs:
                    self.significant_features_csvs.append(_significant_filepath)
                for classifier_model in self.classifier_models:
                    classifier_slug = classifier_model.slug_name
                    records_per_classifier[classifier_slug].append(
                        {
                            "random_state": _rs,
                            "significant_features_csv": _significant_filepath,
                            "trained_model_filepath": _model_filepaths[classifier_slug],
                        }
                    )
            else:
                pending_jobs.append(
                    (
                        _rs,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )

        return pending_jobs, records_per_classifier

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train on the full dataset across multiple seeds (no train/test split).

        For each seed, performs:

        1. Random under-sampling on full dataset (ONCE per seed)
        2. Feature selection on resampled data (ONCE per seed, shared across classifiers)
        3. For each classifier: training with GridSearchCV
        4. For each classifier: save trained model

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

        # Pre-filter: collect already-completed seeds without re-running them
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        pending_jobs, records_per_classifier = self._collect_pending_train_jobs(
            random_states,
            sampling_strategy,
            save_all_features,
            plot_significant_features,
        )

        for seed_results in self._run_jobs(self._run_train, pending_jobs):
            if seed_results is None:  # Feature selection returned nothing
                continue
            # seed_results is dict[classifier_slug -> 3-tuple]
            for classifier_slug, (
                _random_state,
                significant_features_csv,
                trained_model_filepath,
            ) in seed_results.items():
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
