import os
from typing import Self, Literal
from datetime import datetime
from functools import cached_property
from collections.abc import Callable

import numpy as np
import joblib
import pandas as pd
from joblib import Parallel, delayed

from eruption_forecast import (
    TremorData,
    LabelBuilder,
    FeaturesBuilder,
    DynamicLabelBuilder,
    TremorMatrixBuilder,
)
from eruption_forecast.plots import plot_significant_features
from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import (
    resample,
    grid_search_cv,
    save_model_registry,
    get_classifier_models,
)
from eruption_forecast.utils.dataframe import concat_significant_features
from eruption_forecast.utils.pathutils import ensure_dir, generate_features_filepaths
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


class TrainingModel(BaseModel):
    """TrainingModel"""

    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        classifiers: str | list[str],
        eruption_dates: list[str],
        window_size: int = 2,
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified"
        ] = "shuffle-stratified",
        cv_splits: int = 5,
        number_of_features: int = 20,
        include_eruption_date: bool = False,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        n_grids: int = 1,
        verbose: bool = False,
    ) -> None:
        # Set properties
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            eruption_dates=eruption_dates,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        classifiers = [classifiers] if isinstance(classifiers, str) else classifiers

        # Set default properties
        self._tremor_data = tremor_data
        self.classifiers = classifiers
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_features = number_of_features
        self.include_eruption_date: bool = include_eruption_date
        self.overwrite: bool = overwrite
        self.n_grids: int = n_grids

        # Set additional properties
        self.classifier_models: list[ClassifierModel] = get_classifier_models(
            classifiers,
            cv_strategy=self.cv_strategy,
            cv_splits=self.cv_splits,
            verbose=verbose,
        )
        self.cv_name = self.classifier_models[0].slug_cv_name
        self.FeatureSelector = FeatureSelector(
            method="tsfresh",
            n_jobs=n_grids,
            verbose=verbose,
        )

        (
            self.training_dir,
            self.features_dir,
            self.features_seed_dir,
            self.features_resampled_dir,
            self.figures_seed_dir,
            self.classifier_dirs,
            self.models_dir,
        ) = self.set_directories()
        self.features_csvs: list[str] = []
        self.features_selected_df: pd.DataFrame = pd.DataFrame()

        # Will be set after build_Label() called
        self.LabelBuilder: LabelBuilder | None = None
        self.labels: pd.Series = pd.Series()
        self.basename: str | None = None

        # WIll be set after extract_features() called
        self.extracted_features_df: pd.DataFrame = pd.DataFrame()

        # WIll be set after fit() called
        self.csv: dict[str, str] = {}

        self.validate()

    @cached_property
    def tremor_data(self) -> TremorData:
        if not isinstance(self._tremor_data, str | pd.DataFrame):
            raise TypeError(
                f"tremor_data should have an instance of `str` or `pd.DataFramae` "
                f"instead of {type(self._tremor_data)}"
            )

        if isinstance(self._tremor_data, str):
            return TremorData.from_csv(self._tremor_data)

        return TremorData(self._tremor_data)

    def set_directories(self):
        training_dir = os.path.join(self.output_dir, "training")
        features_dir = os.path.join(training_dir, "features", self.cv_name)
        features_seed_dir = os.path.join(features_dir, "seed")
        features_resampled_dir = os.path.join(features_dir, "resampled")
        figures_seed_dir = os.path.join(features_seed_dir, "figures")

        classifier_dirs: dict[str, str] = {}
        models_dir: dict[str, str] = {}

        for classifier_model in self.classifier_models:
            classifier_slug_name = classifier_model.slug_name
            classifier_dir = os.path.join(
                training_dir,
                "classifiers",
                classifier_slug_name,
                self.cv_name,
            )
            classifier_dirs[classifier_slug_name] = classifier_dir
            model_dir = os.path.join(classifier_dir, "models")
            ensure_dir(classifier_dir)
            models_dir[classifier_slug_name] = model_dir

        return (
            training_dir,
            features_dir,
            features_seed_dir,
            features_resampled_dir,
            figures_seed_dir,
            classifier_dirs,
            models_dir,
        )

    def validate(self) -> Self:
        """Validate the model parameters."""

        # Ensure total grid not over than total CPU
        total_grid = self.n_jobs * self.n_grids
        if total_grid > self.total_cpu:
            self.n_grids = np.clip(self.total_cpu // self.n_jobs, 1, self.total_cpu)

        # Optimize n_grids search to utitlize all available CPU
        if self.n_jobs == 1 and self.n_grids == 1:
            self.n_grids = self.total_cpu - 2

        # Ensuring training dates under tremor dates
        tremor_data: TremorData = self.tremor_data
        tremor_start_date = tremor_data.start_date
        tremor_end_date = tremor_data.end_date
        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            logger.info(
                f"Training start date adjusted to tremor start date: "
                f"{tremor_start_date.strftime('%Y-%m-%d')}"
            )
        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            logger.info(
                f"Training end date adjusted to tremor end date: "
                f"{tremor_end_date.strftime('%Y-%m-%d')}"
            )

        ensure_dir(self.training_dir)

        return self

    def describe(self) -> str:
        return "describe"

    def to_dict(self) -> dict:
        result: dict = {
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "eruption_dates": self.eruption_dates,
            "n_jobs": self.n_jobs,
        }

        return result

    def to_prompt(self) -> str:
        return "to_prompt"

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Instantiate and build a label builder of the requested type.

        Args:
            window_step (int): Window size in days for training data windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            builder (Literal["standard", "dynamic"]): Label builder variant.
                ``"standard"`` uses a single global window; ``"dynamic"``
                generates one window per eruption event.
            days_before_eruption (int | None): Days before each eruption to
                start its window. Required when ``builder="dynamic"``.
                Defaults to None.
            verbose (bool | None): Override self.verbose. Defaults to None.
        """
        if window_step <= 0:
            raise ValueError("window_step (in day) must be > 0.")

        verbose = verbose if verbose is not None else self.verbose

        if builder == "dynamic":
            if days_before_eruption is None:
                raise ValueError(
                    "days_before_eruption is required when builder='dynamic'."
                )

            label_builder = DynamicLabelBuilder(
                days_before_eruption=days_before_eruption,
                window_step=window_step,
                window_step_unit=window_step_unit,
                day_to_forecast=self.window_size,
                eruption_dates=self.eruption_dates,  # ty:ignore[invalid-argument-type]
                output_dir=self.training_dir,
                root_dir=self.root_dir,
                verbose=verbose,
            ).build()
        else:
            if days_before_eruption:
                logger.info(
                    "Using standart label builder, ``days_before_eruption`` will be ignored."
                )

            label_builder = LabelBuilder(
                start_date=self.start_date,
                end_date=self.end_date,
                window_step=window_step,
                window_step_unit=window_step_unit,
                day_to_forecast=self.window_size,
                eruption_dates=self.eruption_dates,  # ty:ignore[invalid-argument-type]
                output_dir=self.training_dir,
                root_dir=self.root_dir,
                verbose=verbose,
            ).build()

        self.LabelBuilder = label_builder
        self.basename = os.path.basename(label_builder.csv).split(".csv")[0]

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        save_tremor_matrix_per_id: bool = False,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        if self.LabelBuilder is None:
            raise ValueError("Please run build_label() first.")

        verbose = verbose if verbose is not None else self.verbose

        tremor_matrix_df = (
            TremorMatrixBuilder(
                tremor_df=self.tremor_data.df,
                label_df=self.LabelBuilder.df,
                output_dir=os.path.join(self.training_dir, "tremor"),
                window_size=self.window_size,
                overwrite=overwrite or self.overwrite,
                verbose=verbose,
            )
            .build(
                select_tremor_columns=select_tremor_columns,
                save_tremor_matrix_per_method=save_tremor_matrix_per_method,
                save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            )
            .df
        )

        features_builder = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            label_df=self.LabelBuilder.df,
            label_features_basename=self.basename,
            output_dir=self.features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=verbose,
        )

        self.extracted_features_df = features_builder.extract_features(
            use_relevant_features=True,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        labels = features_builder.label_df.copy()
        if "id" in labels.columns:
            labels = labels.set_index("id")
        if "datetime" in labels.columns:
            labels = labels.drop("datetime", axis=1)

        self.labels = labels["is_erupted"]

        return self

    def create_directories(
        self,
        plot_features: bool = False,
    ) -> None:
        ensure_dir(self.training_dir)
        ensure_dir(self.features_dir)
        ensure_dir(self.features_seed_dir)
        ensure_dir(self.features_resampled_dir)

        for model_name in self.models_dir.keys():
            ensure_dir(self.models_dir[model_name])

        if plot_features:
            ensure_dir(self.figures_seed_dir)

    def fit(
        self,
        seeds: int = 25,
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = False,
    ) -> Self:
        """Train on the full dataset across multiple seeds (no train/test split)."""
        if self.LabelBuilder is None:
            raise ValueError("Please run build_label() first.")

        if self.extracted_features_df.empty:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories(plot_features=plot_features)

        if resample_method == "auto":
            minority_share = self.LabelBuilder.df.value_counts(normalize=True).min()
            if minority_share <= minority_threshold:
                resample_method = "under"
                logger.info(
                    f"resample_method='auto': minority class is {minority_share:.1%} "
                    "(<10%) — using 'under' (RandomUnderSampler)."
                )
            else:
                resample_method = None
                logger.info(
                    f"resample_method='auto': minority class is {minority_share:.1%} "
                    f"(>{minority_threshold * 100}%) — skipping resampling."
                )

        random_states: list[int] = list(range(seeds))
        (
            pending_feature_selection_jobs,
            pending_training_model_jobs,
            records_per_classifier,
        ) = self._collect_pending_train_jobs(
            random_states=random_states,
            resample_method=resample_method,
            sampling_strategy=sampling_strategy,
            plot_features=plot_features,
        )

        logger.info(f"Running parallel feature selection across {seeds} seeds..")

        feature_selection_results: list[str | None] = self._run_jobs(
            self._features_selection,
            pending_feature_selection_jobs,
            job_name="Pending Feature Selection",
        )

        new_training_model_jobs: list[tuple] = []

        for feature_selection_result, pending_feature_selection_job in zip(
            feature_selection_results, pending_feature_selection_jobs, strict=True
        ):
            if feature_selection_result is None:
                continue
            _random_state = pending_feature_selection_job[0]
            for classifier_model in self.classifier_models:
                new_training_model_jobs.append(
                    (_random_state, classifier_model.slug_name)
                )

        all_training_model_jobs = pending_training_model_jobs + new_training_model_jobs
        training_model_results: list[str | None] = self._run_jobs(
            self._train,
            all_training_model_jobs,
            job_name="Pending Feature Selection",
        )

        for result in training_model_results:
            if result is None:
                continue

            classifier_slug, _random_state, features_seed_path, model_seed_path = result

            if classifier_slug not in records_per_classifier:
                records_per_classifier[classifier_slug] = []

            if features_seed_path not in self.features_csvs:
                self.features_csvs.append(features_seed_path)

            records_per_classifier[classifier_slug].append(
                {
                    "random_state": _random_state,
                    "features_csv": features_seed_path,
                    "model_filepath": model_seed_path,
                }
            )

        for _rs in random_states:
            filename = f"{_rs:05d}"
            sf = os.path.join(self.features_seed_dir, f"{filename}.csv")
            if os.path.isfile(sf) and sf not in self.features_csvs:
                self.features_csvs.append(sf)

        if self.verbose:
            logger.info("Prediction: Concatenating significant features")

        self.features_selected_df = concat_significant_features(
            features_csvs=self.features_csvs,
            features_dir=self.features_dir,
            number_of_features=self.number_of_features,
        )

        if plot_features and not self.features_selected_df.empty:
            plot_significant_features(
                df=self.features_selected_df.reset_index(),
                filepath=os.path.join(self.features_dir, f"top_{self.number_of_features}_features"),
                overwrite=True,
                values_column="score",
            )

        # Save registry per classifier
        for classifier_model in self.classifier_models:
            if self.verbose:
                logger.info(f"Prediction: Saving {classifier_model} model registry...")

            classifier_slug = classifier_model.slug_name
            if not records_per_classifier[classifier_slug]:
                continue
            self.csv[classifier_slug] = save_model_registry(
                seeds=seeds,
                records=records_per_classifier[classifier_slug],
                classifier_dir=self.classifier_dirs[classifier_slug],
                classifier_model=classifier_model,
                number_of_features=self.number_of_features,
            )

        if self.verbose:
            logger.info(f"Prediction: Models saved to: {self.csv}")

        return self

    def _train(
        self,
        random_state: int,
        classifier_slug: str,
    ) -> tuple | None:
        filename = f"{random_state:05d}"
        features_seed_path = os.path.join(self.features_seed_dir, f"{filename}.csv")
        features_resampled_path = os.path.join(
            self.features_resampled_dir, f"{filename}.csv"
        )

        if not os.path.isfile(features_seed_path):
            logger.warning(
                f"Features {random_state:05d}: {features_seed_path} missing, skipping."
            )
            return None

        top_n_features = pd.read_csv(features_seed_path, index_col=0).index.tolist()
        if not top_n_features:
            return None

        classifier_model = next(
            m for m in self.classifier_models if m.slug_name == classifier_slug
        )

        model_seed_path = os.path.join(
            self.models_dir[classifier_slug], f"{filename}.pkl"
        )

        # Return cached result without re-training if the model already exists.
        if not self.overwrite and os.path.isfile(model_seed_path):
            logger.info(
                f"Seed {random_state:05d} / {classifier_slug}: model exists, skipping."
            )
            return classifier_slug, random_state, features_seed_path, model_seed_path

        _resampled_df = pd.read_csv(features_resampled_path, index_col=0)
        labels_resampled = _resampled_df["is_erupted"]
        features_resampled = _resampled_df.drop(columns=["is_erupted"])

        if self.verbose:
            logger.info(f"Fitting Seed: {random_state:05d} / {classifier_slug}...")

        _, _, best_model = grid_search_cv(
            random_state,
            features_resampled,
            labels_resampled,
            top_n_features,
            classifier_model=classifier_model,
        )

        joblib.dump(best_model, model_seed_path)

        if self.verbose:
            logger.info(
                f"Fitted Model {random_state:05d} / {classifier_slug} : {model_seed_path}"
            )

        return classifier_slug, random_state, features_seed_path, model_seed_path

    def _features_selection(
        self,
        random_state: int,
        features_seed_path: str,
        features_resampled_path: str,
        figures_seed_path: str | None,
        resample_method: Literal["under", "over"] | None,
        sampling_strategy: str | float,
    ) -> str | None:
        features_resampled, labels_resampled = resample(
            features=self.extracted_features_df,
            labels=self.labels,
            method=resample_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            verbose=self.verbose,
        )

        pd.concat([features_resampled, labels_resampled], axis=1).to_csv(
            features_resampled_path
        )

        result = self._select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            number_of_features=self.number_of_features,
            features_seed_path=features_seed_path,
            figures_seed_path=figures_seed_path,
        )

        if result is None:
            return None

        return features_seed_path

    def _run_jobs(
        self,
        method: Callable,
        jobs: list[tuple],
        job_name: str = "Training Model",
    ) -> list:
        if self.n_jobs != 1:
            logger.info(
                f"[{job_name}]: Running on {self.n_jobs} job(s). Grid search jobs {self.n_grids}..."
            )
            return Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(method)(*job) for job in jobs
            )
        return [method(*job) for job in jobs]

    def _collect_pending_train_jobs(
        self,
        random_states: list[int],
        resample_method: Literal["under", "over", "auto"] | None,
        sampling_strategy: str | float,
        plot_features: bool,
    ) -> tuple[list[tuple], list[tuple], dict[str, list[dict]]]:
        pending_feature_selection_jobs: list[tuple] = []
        pending_training_model_jobs: list[tuple] = []

        records_per_classifier: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }

        for random_state in random_states:
            (
                can_skip,
                features_seed_path,
                features_resampled_path,
                figures_seed_path,
            ) = generate_features_filepaths(
                random_state=random_state,
                features_seed_dir=self.features_seed_dir,
                features_resampled_dir=self.features_resampled_dir,
                figures_seed_dir=self.figures_seed_dir,
                plot_features=plot_features,
                overwrite=self.overwrite,
            )

            if can_skip:
                continue

            pending_feature_selection_jobs.append(
                (
                    random_state,
                    features_seed_path,
                    features_resampled_path,
                    figures_seed_path,
                    resample_method,
                    sampling_strategy,
                )
            )

            filename = f"{random_state:05d}"
            for classifier_model in self.classifier_models:
                classifier_slug = classifier_model.slug_name
                model_seed_path = os.path.join(
                    self.models_dir[classifier_slug], f"{filename}.pkl"
                )

                if not self.overwrite and os.path.isfile(model_seed_path):
                    records_per_classifier[classifier_slug].append(
                        {
                            "random_state": random_state,
                            "features_csv": features_seed_path,
                            "model_filepath": model_seed_path,
                        }
                    )
                else:
                    pending_training_model_jobs.append((random_state, classifier_slug))

        return (
            pending_feature_selection_jobs,
            pending_training_model_jobs,
            records_per_classifier,
        )

    def _select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        random_state: int,
        features_seed_path: str,
        number_of_features: int,
        figures_seed_path: str | None,
        overwrite: bool = False,
    ):
        # Reduced features/columns
        features_selector = self.FeatureSelector.set_random_state(random_state)

        df_selected_features = features_selector.fit_transform(
            features, labels, top_n=number_of_features
        )

        if features_selector.n_features == 0:
            logger.warning(
                f"{random_state:05d}: Features reduced to 0. Skip training model."
            )
            return None

        selected_features = features_selector.selected_features_

        # Handle if columns in df_selected_features has less than number_of_significant_features
        len_features_columns = len(df_selected_features.columns)
        if len_features_columns < number_of_features:
            logger.warning(
                f"{random_state:05d}: Number of features after extracted ({len_features_columns}) "
                f"are less than {number_of_features} features."
            )
            number_of_features = len_features_columns

        top_selected_features = selected_features.head(number_of_features)

        # Save TOP-N significant features
        top_selected_features.to_csv(features_seed_path, index=True)

        if figures_seed_path is not None:
            plot_significant_features(
                df=pd.DataFrame(selected_features).reset_index(),
                filepath=figures_seed_path,
                top_features=number_of_features,
                values_column="score",
                overwrite=overwrite,
                dpi=150,
            )

        return (
            df_selected_features,
            top_selected_features,
            selected_features,
            number_of_features,
        )
