import os
import multiprocessing
from abc import ABC, abstractmethod
from typing import Self, Literal
from datetime import datetime
from functools import cached_property

import pandas as pd

from eruption_forecast import (
    TremorData,
    LabelBuilder,
    FeaturesBuilder,
    DynamicLabelBuilder,
    TremorMatrixBuilder,
)
from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import sort_dates, to_datetime


class BaseModel(ABC):
    """Abstract base class for eruption forecast model stages.

    Attributes:
        start_date (datetime): Inclusive start of the modelling period.
        end_date (datetime): Inclusive end of the modelling period.
        window_size (int): Sliding window size in days.
        eruption_dates (list[datetime] | None): Sorted eruption datetimes, or
            ``None`` when no eruption dates are provided.
        output_dir (str): Resolved path to the output directory.
        root_dir (str | None): Optional project root used to resolve
            ``output_dir``.
        n_jobs (int): Number of parallel workers.
        verbose (bool): Whether to emit verbose log messages.
        start_date_str (str): ``start_date`` formatted as ``YYYY-MM-DD``.
        end_date_str (str): ``end_date`` formatted as ``YYYY-MM-DD``.
        n_days (int): Total number of days in the modelling period.
        total_cpu (int): Number of logical CPUs on the current machine.
    """

    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        eruption_dates: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initializes the base model.

        Args:
            tremor_data (str | DataFrame): Path to a tremor CSV file or a
                pre-loaded DataFrame.
            start_date (str | datetime): Inclusive start of the modelling
                period.
            end_date (str | datetime): Inclusive end of the modelling period.
            window_size (int, optional): Sliding window size in days. Defaults
                to 2.
            eruption_dates (list[str] | None, optional): Known eruption dates
                in ``YYYY-MM-DD`` format. Defaults to ``None``.
            output_dir (str | None, optional): Output directory path. Resolved
                relative to ``root_dir`` when omitted. Defaults to ``None``.
            root_dir (str | None, optional): Project root directory used to
                resolve ``output_dir``. Defaults to ``None``.
            n_jobs (int, optional): Number of parallel workers. Capped at
                ``cpu_count - 2`` when too high. Defaults to 1.
            verbose (bool, optional): Emit verbose log messages when ``True``.
                Defaults to ``False``.
        """

        # Set properties
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )

        # Set default properties
        self._tremor_data = tremor_data
        self.start_date = to_datetime(start_date)
        self.end_date = to_datetime(end_date)
        self.window_size = window_size
        self.eruption_dates = sort_dates(eruption_dates) if eruption_dates else None
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set additional properties
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.n_days = (self.end_date - self.start_date).days + 1
        self.total_cpu = multiprocessing.cpu_count()

        self._validate()

    def _validate(self) -> Self:
        """Validates the base model parameters and clamps ``n_jobs``.

        Returns:
            Self: The current instance, for method chaining.
        """
        validate_date_ranges(self.start_date, self.end_date)

        if self.n_jobs >= self.total_cpu:
            n_jobs = self.total_cpu - 2
            if self.verbose:
                logger.warning(
                    f"Value of n_jobs ({self.n_jobs}) is more than {self.total_cpu} CPU. "
                    f"Update n_jobs to {n_jobs} CPU."
                )
            self.n_jobs = n_jobs

        return self

    def _sync_dates_to_tremor(self) -> None:
        """Clamps ``start_date`` and ``end_date`` to the tremor data range.

        Adjusts ``self.start_date`` and ``self.end_date`` when they fall
        outside the span of the loaded tremor CSV. Called lazily from
        ``build_label()`` to avoid triggering a CSV read at construction time.
        """
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

    @cached_property
    def tremor_data(self) -> TremorData:
        """The validated ``TremorData`` instance loaded from the configured source.

        Accepts a filesystem path or a pre-loaded DataFrame. The result is
        cached after the first access so subsequent reads are free.

        Returns:
            TremorData: Validated tremor data container.

        Raises:
            TypeError: If ``_tremor_data`` is neither a ``str`` nor a
                ``DataFrame``.
        """
        if not isinstance(self._tremor_data, str | pd.DataFrame):
            raise TypeError(
                f"tremor_data should have an instance of `str` or `pd.DataFrame` "
                f"instead of {type(self._tremor_data)}"
            )

        if isinstance(self._tremor_data, str):
            return TremorData.from_csv(self._tremor_data)

        return TremorData(self._tremor_data)

    @abstractmethod
    def set_directories(self) -> tuple:
        """Build and return all directory paths needed for class."""

    @abstractmethod
    def create_directories(self) -> None:
        """Creates the output directories required by this model."""

    @abstractmethod
    def validate(self) -> Self:
        """Validates model-specific parameters."""

    @abstractmethod
    def describe(self) -> str:
        """Returns a human-readable prose description of this instance."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns a structured dictionary of all configuration and derived fields."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Returns a structured text block suitable for LLM and MCP prompt input.

        Builds a deterministic, template-driven bullet-list block from
        ``to_dict()`` that is stable across calls. Paths are reduced to
        filenames only to avoid leaking absolute system paths into prompts.
        For human-readable prose use ``describe()``. For raw data use
        ``to_dict()``.

        Returns:
            str: Formatted bullet-list block ready for MCP tool responses or
                LLM context.
        """

    @abstractmethod
    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Builds the label set for this model.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            Self: The current instance, for method chaining.
        """

    @abstractmethod
    def extract_features(self) -> Self:
        """Extracts features from the tremor data.

        Returns:
            Self: The current instance, for method chaining.
        """
