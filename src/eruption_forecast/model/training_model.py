import os
from typing import Self, Literal
from datetime import datetime

import numpy as np
import pandas as pd
from stack_data.utils import cached_property

from eruption_forecast import (
    TremorData,
    LabelBuilder,
    FeaturesBuilder,
    DynamicLabelBuilder,
    TremorMatrixBuilder,
)
from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import get_classifier_models
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.classifier_model import ClassifierModel


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
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified"
        ] = "shuffle-stratified",
        cv_splits: int = 5,
        minority_threshold: float = 0.15,
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
        self.resample_method = resample_method
        self.minority_threshold = minority_threshold
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

        (
            self.training_dir,
            self.features_dir,
            self.significant_features_dir,
            self.figures_dir,
        ) = self.set_directories()

        # Will be set after build_Label() called
        self.LabelBuilder: LabelBuilder | None = None
        self.basename: str | None = None

        # WIll be set after extract_features() called
        self.features_df: pd.DataFrame = pd.DataFrame()

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
        significant_features_dir = os.path.join(features_dir, "significant")
        figures_dir = os.path.join(significant_features_dir, "figures")

        return (
            training_dir,
            features_dir,
            significant_features_dir,
            figures_dir,
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
        output_dir: str | None = None,
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
            output_dir (str | None): Output directory for data labelling.
            verbose (bool | None): Override self.verbose. Defaults to None.
        """
        if window_step <= 0:
            raise ValueError("window_step (in day) must be > 0.")

        verbose = verbose if verbose is not None else self.verbose
        output_dir = output_dir or self.training_dir
        ensure_dir(output_dir)

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
                output_dir=output_dir,
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
                output_dir=output_dir,
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
        save_tremor_matrix_per_id: bool = False,
        exclude_features: list[str] | None = None,
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

        self.features_df = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            label_df=self.LabelBuilder.df,
            label_features_basename=self.basename,
            output_dir=self.features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=verbose,
        ).extract_features(
            use_relevant_features=True,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        return self

    def create_directories(
        self,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        **kwargs,
    ) -> None:
        ensure_dir(self.training_dir)

    def train(
        self,
        seeds=25,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> Self:
        random_states: list[int] = list(range(seeds))

        # Ensure directories

        return self

    def evaluate(self) -> Self:
        return self

    def fit(self, with_evaluation: bool = False, **kwargs) -> Self:
        """Dispatch to ``evaluate()`` or ``train()`` based on ``with_evaluation``.

        Args:
            with_evaluation (bool, optional): If True, calls ``evaluate()``
                (80/20 split + metrics). If False, calls ``train()`` (full dataset,
                no metrics). Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the chosen method.

        Returns:
            Self: The ModelTrainer instance for method chaining.
        """
        self.train(**kwargs)

        if with_evaluation:
            self.evaluate(**kwargs)

        return self
