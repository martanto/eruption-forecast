import os
from typing import Any, Self, Literal
from datetime import datetime

import pandas as pd
import matplotlib

from eruption_forecast.plots import plot_forecast
from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.utils.date_utils import set_datetime_index
from eruption_forecast.utils.formatting import pdf_metadata
from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble


matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

        self.ClassifierEnsemble: ClassifierEnsemble = (
            model
            if isinstance(model, ClassifierEnsemble)
            else ClassifierEnsemble.from_any(model, verbose)
        )
        self.overwrite = overwrite
        self.basename = f"{self.start_date_str}_{self.end_date_str}"
        (
            self.prediction_dir,
            self.features_dir,
            self.result_dir,
        ) = self.set_directories()

        self.labels: pd.DataFrame = pd.DataFrame()
        self.df: pd.DataFrame = pd.DataFrame()

    def set_directories(self) -> tuple[str, str, str]:
        prediction_dir = os.path.join(self.output_dir, "prediction")
        features_dir = os.path.join(prediction_dir, "features")
        result_dir = os.path.join(prediction_dir, "results")

        return prediction_dir, features_dir, result_dir

    def create_directories(self) -> None:
        ensure_dir(self.prediction_dir)
        ensure_dir(self.features_dir)
        ensure_dir(self.result_dir)

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
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_csv = features_builder.csv

        return self

    def forecast(
        self,
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        **plot_kwargs,
    ) -> pd.DataFrame:
        if self.labels.empty:
            raise ValueError("Please run build_label() first.")

        if self.features_df.empty and self.features_csv is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories()

        results = self.ClassifierEnsemble.predict_with_uncertainty(
            X=self.features_df,
            save=save_seed_result,
            output_dir=self.result_dir,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

        df_forecast = pd.DataFrame(results, index=self.features_df.index)
        csv_path = os.path.join(
            self.output_dir, f"result_all_model_predictions_{self.basename}.csv"
        )

        if not self.labels.empty:
            df_forecast = set_datetime_index(self.labels, df_forecast)

        df_forecast.to_csv(csv_path)
        logger.info(f"Predictions saved to: {csv_path}")

        self._plot_forecast(
            df_forecast,
            plot_threshold,
            title=plot_title,
            plot_pdf=plot_pdf,
            **plot_kwargs,
        )

        return df_forecast

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        threshold: float,
        title: str | None = None,
        plot_pdf: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        if self.labels.empty:
            raise ValueError("No labels dataframe provided.")

        fig = plot_forecast(
            df=df,
            label_df=self.labels,
            threshold=threshold,
            title=title,
            **plot_kwargs,
        )

        figure_dir = os.path.join(self.prediction_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)

        path = os.path.join(figure_dir, f"forecast_{self.basename}.png")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor=None
        )

        logger.info(f"Forecast plot saved to: {path}")

        if plot_pdf:
            path = os.path.join(figure_dir, f"forecast_{self.basename}.pdf")

            # Type 42 embeds TrueType fonts — text stays selectable and
            # renders consistently in all PDF viewers and vector editors.
            with matplotlib.rc_context({"pdf.fonttype": 42}):
                fig.savefig(
                    path,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor=None,
                    metadata=pdf_metadata(
                        title=f"Eruption Forecast: {self.start_date_str} to {self.end_date_str}"
                    ),
                )

        plt.close(fig)
        self.forecast_plot_path = path
