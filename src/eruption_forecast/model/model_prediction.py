import os
from typing import Self
from datetime import datetime

from eruption_forecast.model.base_model import BaseModel


class ModelPrediction(BaseModel):
    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            start_date,
            end_date,
            "predictions",
            output_dir,
            root_dir,
            n_jobs,
            overwrite,
            verbose,
        )

    def build_label(self) -> Self:
        return self

    def extract_features(self) -> Self:
        return self
