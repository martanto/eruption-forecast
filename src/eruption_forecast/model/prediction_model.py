import os
from typing import Self, Literal
from datetime import datetime

import pandas as pd

from eruption_forecast.model import SeedEnsemble, ClassifierEnsemble
from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel


class PredictionModel(BaseModel):
    """Forecast eruption using trained model.

    Loads models produced by ``TrainingModel.fit()`` and runs forecast
    inference. Supports single-model or multi-model consensus predictions by
    aggregating across classifiers and seeds with uncertainty quantification.
    Use :meth:`predict_proba` for unlabelled forecast mode.

    Args:
        model (str | ClassifierEnsemble | SeedEnsemble): Trained model source.
            Accepted forms:

            - ``ClassifierEnsemble`` object — used directly.
            - ``SeedEnsemble`` object — wrapped into a ``ClassifierEnsemble``.
            - Path to ``ClassifierEnsemble.json`` (from ``TrainingModel``).
            - Path to ``ClassifierEnsemble.pkl`` or ``SeedEnsemble_*.pkl``.
            - Path to a trained-model registry ``*.csv`` (one value from
              ``TrainingModel.results``).
    """

    def __init__(
        self,
        model: str | ClassifierEnsemble | SeedEnsemble,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            eruption_dates=None,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.model: ClassifierEnsemble = (
            model
            if isinstance(model, ClassifierEnsemble)
            else ClassifierEnsemble.from_any(model, verbose)
        )
        self.overwrite = overwrite
        self.basename = f"{self.start_date_str}_{self.end_date_str}"
        self.prediction_dir, self.features_dir = self.set_directories()

        self.labels: pd.DataFrame = pd.DataFrame()

    def set_directories(self) -> tuple[str, str]:
        prediction_dir = os.path.join(self.output_dir, "prediction")
        features_dir = os.path.join(prediction_dir, "features")

        return prediction_dir, features_dir

    def create_directories(self) -> None:
        ensure_dir(self.prediction_dir)
        ensure_dir(self.features_dir)

    def validate(self) -> Self:
        return self

    def describe(self) -> str:
        return ""

    def to_dict(self) -> dict:
        return {}

    def to_prompt(self) -> str:
        return ""

    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        if window_step <= 0:
            raise ValueError("window_step must be > 0.")

        filename = (
            f"features-label_{self.basename}_step-{window_step}-{window_step_unit}"
        )
        label_csv = os.path.join(self.features_dir, f"{filename}.csv")
        ensure_dir(self.features_dir)

        if os.path.exists(label_csv) and not self.overwrite:
            self.labels = pd.read_csv(label_csv, index_col=0, parse_dates=True)
            return self

        label_df = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )
        label_df["id"] = range(label_df.shape[0])
        self.labels = label_df

        label_df.to_csv(label_csv, index=True)

        if self.verbose:
            logger.info(f"Label for prediction: {label_csv}")

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        if self.labels.empty:
            raise ValueError("Please run build_label() first.")

        if not self.features_df.empty and self.features_csv is not None:
            if self.verbose:
                logger.info(f"Features already extracted: {self.features_csv}")
            return self

        features_builder = self._build_features(
            label_df=self.labels,
            output_dir=self.prediction_dir,
            features_dir=self.features_dir,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=False,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.features_df = features_builder.extract_features(
            use_relevant_features=False,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_csv = features_builder.csv

        return self

    def predict_proba(self) -> pd.DataFrame:
        if self.labels.empty:
            raise ValueError("Please run build_label() first.")

        if self.features_df.empty and self.features_csv is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories()

        return pd.DataFrame()
