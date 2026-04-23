from abc import ABC, abstractmethod
from typing import Self
from datetime import datetime

import pandas as pd

from eruption_forecast.dataclass import ModelData
from eruption_forecast.utils.pathutils import ensure_dir


class BaseModel(ABC):
    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        sub_dir: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        model = ModelData(
            _tremor_data=tremor_data,
            _start_date=start_date,
            _end_date=end_date,
            _sub_dir=sub_dir,
            _output_dir=output_dir,
            _root_dir=root_dir,
        )

        self.start_date: datetime = model.start_date
        self.end_date: datetime = model.end_date
        self.start_date_str: str = model.start_date_str
        self.end_date_str: str = model.end_date_str
        self.output_dir: str = model.output_dir
        self.root_dir: str | None = root_dir
        self.n_jobs: int = n_jobs
        self.overwrite: bool = overwrite
        self.verbose: bool = verbose

    def create_directories(self) -> None:
        ensure_dir(self.output_dir)

    @abstractmethod
    def build_tremor_matrix(self, *args, **kwargs) -> Self:
        """Build tremor matrix from tremor dataframe and labels"""

    @abstractmethod
    def build_label(self, *args, **kwargs) -> Self:
        """Build label for this model"""

    @abstractmethod
    def extract_features(self, *args, **kwargs):
        """Extract features from this model"""
