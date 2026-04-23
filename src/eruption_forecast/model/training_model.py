from typing import Any, Self, Literal
from datetime import datetime

import pandas as pd

from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder


class TrainingModel(BaseModel):
    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        tremor_columns: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            tremor_data,
            start_date,
            end_date,
            "training",
            output_dir,
            root_dir,
            n_jobs,
            overwrite,
            verbose,
        )

        self.tremor_columns = tremor_columns

        self.label_builder: LabelBuilder | None = None

    def build_tremor_matrix(self) -> Self:
        return self

    def build_label(
        self,
        volcano_id: str,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        include_eruption_date: bool = False,
        output_dir: str | None = None,
        builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
    ) -> Self:
        kwargs: dict[str, Any] = {
            "volcano_id": volcano_id,
            "window_step": window_step,
            "window_step_unit": window_step_unit,
            "day_to_forecast": day_to_forecast,
            "eruption_dates": eruption_dates,
            "include_eruption_date": include_eruption_date,
            "output_dir": output_dir,
            "root_dir": self.root_dir,
            "verbose": self.verbose,
        }

        if builder == "dynamic":
            if days_before_eruption is None:
                raise ValueError(
                    "days_before_eruption is required when builder='dynamic'."
                )
            self.label_builder = DynamicLabelBuilder(
                days_before_eruption=days_before_eruption,
                **kwargs,
            ).build()

            return self

        elif builder == "standard":
            validate_date_ranges(start_date, end_date)
            self.label_builder = LabelBuilder(
                start_date=to_datetime(start_date),
                end_date=to_datetime(end_date),
                **kwargs,
            ).build()

        else:
            raise ValueError(
                f"Builder {builder} not supported. Please use 'standard' or 'dynamic'."
            )

        return self

    def extract_features(self) -> Self:
        return self

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> None:
        """Train on the full dataset across multiple seeds (no train/test split).

        Args:
            random_state (int, optional): Initial random state seed. Defaults to 0.
            total_seed (int, optional): Total number of seeds to run. Defaults to 500.
            sampling_strategy (str | float, optional): Under-sampling ratio for
                balancing classes. Defaults to 0.75.
            save_features (bool, optional): Save all features per seed,
                not just top-N. Defaults to False.
            plot_features (bool, optional): Generate feature importance
                plots. Defaults to False.
        """
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]

        return None
